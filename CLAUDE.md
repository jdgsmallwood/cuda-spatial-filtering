# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CUDA C++ pipeline for real-time radio-astronomy signal processing: it ingests UDP/PCAP packet
streams from FPGA-based receivers (the "LAMBDA" instrument), correlates and beamforms the data on
the GPU (via the Tensor Core Correlator and ccglib matrix-multiply libraries), applies adaptive
spatial filtering / RFI mitigation via eigendecomposition, and writes results out as
HDF5/PSRDADA/FITS/Redis streams. There's also a Python/marimo toolkit in `scripts/` for offline
analysis of captured PCAP and HDF5 data, and a Textual TUI in `ui/` for configuring/running CMake
builds.

## Build & test

Requires CUDA, cuTENSOR, HDF5, CFITSIO, CasaCore, and PSRDADA (`$PSRHOME` env var must point at a
PSRDADA install). Dependencies in `extern/` are git submodules — run
`git submodule update --init --recursive` if they're missing.

### Building/testing in this docker container (the normal way you'll be run)

The container ships a conda environment at `/opt/conda` alongside system packages, and a few
things need pointing at non-default locations for configure/build/run to all succeed. Use a fresh
build directory (the checked-in `build/` has a stale cache pointing at a no-longer-existent path,
`/tmp/cuda-spatial-filtering` — don't try to reconfigure it; make a new one, e.g. `build_test`):

```bash
mkdir build_test && cd build_test

# CUTENSOR_ROOT: the env var defaults to /usr/lib/x86_64-linux-gnu/libcutensor, which has
# version-numbered subdirs (11/, 11.0/, 12/) and no top-level include/ -- find_cutensor.cmake
# (cmake/find_cutensor.cmake) needs the flat ${CUTENSOR_ROOT}/include + ${CUTENSOR_ROOT}/lib
# layout that /opt/conda happens to have instead:
CUTENSOR_ROOT=/opt/conda cmake -DBUILD_TESTING=ON ..
cmake --build . -- -j$(nproc)

# CUDA_HOME: needed at *runtime* (not just configure time) -- see the NVRTC note below.
export CUDA_HOME=/opt/conda/targets/x86_64-linux
ctest                                    # run everything
ctest -R ProcessorStateTest              # filter by test-suite name (regex)
./tests/CUDASpatialFilteringTests --gtest_filter='*SimpleTest*'   # run a gtest binary directly
```

Two non-obvious things that will otherwise produce confusing failures:

- **Run CUDA/GPU tests outside the Codex sandbox.** The sandbox can block OS/GPU operations that
  these tests need, especially pinned host allocations (`cudaMallocHost`), causing errors like
  `OS call failed or operation not supported on this OS` before any test assertions run. When
  running GPU test binaries or GPU benchmarks from Codex, use an escalated/outside-sandbox command.
- **Link errors like `undefined reference to 'arc4random@GLIBC_2.36'` /
  `strlcpy@GLIBC_2.38'` / `__isoc23_strtoull@GLIBC_2.38'`** when linking any test binary (or
  `apps/*`): the linker resolved `libc.so.6` against an *older* glibc than the system libraries it's
  linking against (`libcurl`, `libpcap`, `gtest`, `libcasa_*`, ...) were built against. The cause:
  `/opt/conda/bin/g++` sits earlier in `$PATH` than `/usr/bin/g++` and defaults to its own bundled
  cross-sysroot glibc (capped at `GLIBC_2.35`, vs. the container's system glibc `2.39`). Fix by
  forcing the link step to resolve against the system glibc — pass this when configuring (it's a
  build-directory/cache setting; don't bake it into the repo's `CMakeLists.txt`, since it's purely
  a function of this container's toolchain layering):
  ```bash
  CUTENSOR_ROOT=/opt/conda cmake -DBUILD_TESTING=ON \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--sysroot=/ -L/usr/lib/x86_64-linux-gnu -Wl,-rpath-link,/usr/lib/x86_64-linux-gnu" \
    ..
  ```
- **`NVRTC_ERROR_COMPILATION` / "cannot open source file `cooperative_groups/memcpy_async.h`"
  at runtime** (not a build failure — the binary links fine and fails when *run*): any test or app
  that constructs a `tcc::Correlator` (i.e. builds a `LambdaGPUPipeline`-family pipeline)
  JIT-compiles `TCCorrelator.cu` via NVRTC at runtime, and NVRTC needs the CUDA toolkit's
  `include/` directory for headers like `cooperative_groups/*.h`.
  `Correlator::findNVRTCincludePath()` (`extern/tcc/libtcc/Correlator.cc`) checks `$CUDA_HOME`,
  then `$CUDA_PATH`, then `$CONDA_PREFIX` (looking for `<prefix>/include/cuda.h` or
  `<prefix>/targets/x86_64-linux/include/cuda.h`) before falling back to introspecting
  `libnvrtc.so`'s load path — so export one of those before running anything that touches the
  correlator, e.g. `export CUDA_HOME=/opt/conda/targets/x86_64-linux`.

(OzStar — not the primary environment, but documented for completeness — instead does
`module load cuda/12.6.0 gcc/13.3.0 cmake/3.29.3`, which keeps the host compiler's glibc and the
system's in sync and avoids both issues above.)

- **`Could NOT find Boost`** (configure-time failure once `extern/gpu-filter` is in the build):
  Boost headers aren't installed anywhere in this container (`libboost-dev` is in the Dockerfile
  for future image rebuilds, but a *running* container built before that lands won't have it, and
  you won't have root to `apt-get install` it). If you can't install the system package, point
  CMake at a headers-only Boost extraction instead (no build/link needed — `gpu-filter`'s
  `FilterBank` only uses header-only `boost::multi_array`):
  ```bash
  cmake ... -DBOOST_ROOT=/path/to/boost-X.Y.Z -DBoost_INCLUDE_DIR=/path/to/boost-X.Y.Z ..
  ```
- **`cuFFTDx requires GPU architecture sm_70 or higher`** at configure time, even though the
  actual GPU is sm_70+: this project's own `CMAKE_CUDA_ARCHITECTURES` default is the string
  `"native"` (set *after* `project(... LANGUAGES CUDA ...)` in the top-level `CMakeLists.txt`, so
  CMake's built-in native-arch auto-resolution never fires and the literal string `"native"`
  survives to configure time). `extern/gpu-filter`'s own architecture check
  (`if(... CMAKE_CUDA_ARCHITECTURES LESS 70)`) does a numeric comparison against that string,
  which is always false-y and trips the guard. Pass an explicit numeric value at configure time
  (this also fixes the "native" issue for every other target, not just gpu-filter):
  ```bash
  cmake ... -DCMAKE_CUDA_ARCHITECTURES=89 ..   # 89 for an RTX 4060; use the actual compute capability
  ```
- **`undefined reference to 'nl_*@libnl_3'` / missing `libnl-route-3.so.200`** when linking any app
  target (not test targets — only `apps/CMakeLists.txt`'s `gpu_app_common` links `IBVERBS_LIBRARY`):
  conda ships its own `libibverbs.so`, built against conda's own *versioned* `libnl-3`/
  `libnl-route-3` — incompatible with this container's *unversioned* system `libnl-3` and (if
  `libnl-route-3-dev` isn't installed) entirely-missing system `libnl-route-3`. The repo's
  `CMakeLists.txt` already prefers a system `libibverbs` (`/usr/lib/x86_64-linux-gnu` searched
  first) and disables `libpcap`'s own independent `pkg_check_modules(libibverbs)` RDMA-sniffing
  probe (`DISABLE_RDMA` — libpcap does this check unconditionally on Linux and would otherwise
  pull conda's copy in a second, independent way even after the app-level preference is fixed).
  This only fully resolves once `libibverbs-dev` + `libnl-route-3-dev` are installed (in the
  Dockerfile for future rebuilds); on a *running* container without them and without root, force
  ibverbs off entirely instead of fighting the conda/system split:
  ```bash
  cmake ... -DIBVERBS_LIBRARY:FILEPATH= -DIBVERBS_INCLUDE_DIR:PATH= ..   # empty (not NOTFOUND) so
                                                                          # find_library doesn't re-search
  ```
- **`ppf::Filter::launchAsync`'s raw-pointer overload (`CUstream`/`CUdeviceptr` args) hangs
  indefinitely in `cuLaunchKernel`**, regardless of CUDA context model — a real bug isolated via
  a minimal repro (just one `Filter` instance, nothing else, not even TCC). `FineChannelizer`
  (`pipeline/common.hpp`) works around it by calling gpu-filter's `cu::Stream`-based overload
  directly, wrapping the raw stream/pointers into non-owning `cu::Stream`/`cu::DeviceMemory`
  objects itself instead of letting `Filter::launchAsync` do it internally. If you're calling
  `ppf::Filter::launchAsync` directly anywhere else, use the `cu::Stream`-based overload, not the
  raw-pointer one. See `docs/architecture.md` Section 6's "Resolved" note for the full
  investigation (including the wrong turns — an earlier theory blamed external GPU contention;
  that was a misdiagnosis of ordinary desktop GPU usage, not the real cause).

Test binaries (see `tests/CMakeLists.txt`): `CUDASpatialFilteringTests` (test_spatial.cpp, mostly
disabled/commented-out correlation experiments), `PipelineTests` (test_pipeline.cu, GPU pipeline
kernels), `CUDASpatialFilteringCPUTests` (test_packet_formats.cpp, CPU-only packet parsing),
`WriterTests` (test_writers.cpp, HDF5 writers), `ProcessorTests` (test_processor.cu, the packet
ring-buffer/`ProcessorState` machinery), `PipelineHarnessSelfTest` (support/test_harness_selftest.cu,
proves out the shared pipeline-test harness in `tests/support/` — see `tests/TESTING.md`),
`FineChannelizerTests` (test_fine_channelizer.cu, exercises `LambdaGPUPipeline` with pre-correlation
fine channelization active, plus direct low-level unit tests of the shared
`reorder_to_filter_input`/`channelizer_output_to_corr_input`/`channelizer_output_to_col_maj_cons`
buffer-layout kernels every pipeline reuses — see the note on GPU-contention-induced startup stalls
below if a run appears to hang), `FineChannelizedAntennaSpectraTests`/
`FineChannelizedBeamformedSpectraTests`/`FineChannelizedProjectionTests`/
`FineChannelizedAdaptiveTests`/`FineChannelizedPulsarFoldTests`/`FineChannelizedCorrBeamTests`
(one per remaining `Lambda*Pipeline` variant, each with a channelized-vs-disabled-config pair of
physical-invariant tests), plus `test_beamforming.cpp` for beamforming math.

Build options that shape the generated code (set with `-D...`, see top-level `CMakeLists.txt`):
`NR_OBSERVING_BUFFERS`, `NR_OBSERVING_FPGA_SOURCES`, `NR_OBSERVING_CHANNELS`,
`NR_OBSERVING_RECEIVERS_PER_PACKET`, `NR_OBSERVING_PADDED_RECEIVERS`,
`NR_OBSERVING_PACKETS_FOR_CORRELATION`, `NR_OBSERVING_CORRELATION_BLOCKS_TO_INTEGRATE`,
`NR_OBSERVING_FINE_CHANNELS`, `NR_OBSERVING_FINE_CHANNEL_EDGE_TRIM`, `NUMBER_BEAMS`. These become
`add_compile_definitions` and feed directly into the `LambdaConfig` template instantiations in each
app's `main()`. They only apply to the `apps` subdirectory build (skipped when `BUILD_TESTING=ON`).
`NR_OBSERVING_FINE_CHANNELS` (default 32) is the pre-correlation PFB channelizer's fine-channel
count per coarse/FPGA channel, via ASTRON's `gpu-filter` (see `docs/architecture.md` Section 6) —
`1` disables channelization entirely, reproducing pre-channelization behaviour exactly.
`NR_OBSERVING_FINE_CHANNEL_EDGE_TRIM` (default 2) is how many fine channels are dropped from each
edge of every coarse channel's fine-channel breakdown before correlation (ignored when
`NR_OBSERVING_FINE_CHANNELS=1`) — the FPGA's own coarse channelizer is a 32/27-oversampled PFB, so
raw samples near a coarse channel's edge carry that channelizer's guard-band aliasing, which
`gpu-filter`'s fine channels nearest those edges inherit; `channelizer_output_to_corr_input`
(`spatial.cuh`) drops them, and `LambdaConfig::NR_CHANNELS` is sized by the trimmed
`NR_EFFECTIVE_FINE_CHANNELS`, not the raw `NR_FINE_CHANNELS`.

The Textual TUI in `ui/` (`cd ui && hatch run ui`, or `python ui.py`) is a convenience wrapper for
setting these CMake cache variables and kicking off a build/configure.

## Architecture

The pipeline is a producer/consumer system glued together by **`LambdaConfig`** — a single struct
template (in `include/spatial/packet_formats.hpp`) that's parameterized over every dimension of the
problem (`NR_CHANNELS`, `NR_FPGA_SOURCES`, `NR_RECEIVERS`, `NR_PACKETS_FOR_CORRELATION`,
`NR_BEAMS`, `NR_PADDED_RECEIVERS`, ...). It defines all the derived array types
(`PacketSamplesType`, `InputPacketSamplesType`, `Visibilities`, `BeamOutput`, etc.) used throughout
the rest of the code as compile-time-shaped multidimensional C arrays. Every app instantiates its
own concrete `Config = LambdaConfig<...>` from the `NR_OBSERVING_*` macros and threads it through
as a template parameter — so most of the "business logic" classes are themselves templates on `T`
(a `LambdaConfig` instantiation), e.g. `ProcessorState<T>`, `LambdaGPUPipeline<T>`,
`SingleHostMemoryOutput<T>`.

Data flow, end to end:

1. **Packet capture** (`PacketInput` hierarchy in `include/spatial/spatial.hpp`,
   implementations in `src/spatial.cpp`): `KernelSocketPacketCapture` (live UDP socket),
   `PCAPPacketCapture` / `PCAPMultiFPGAPacketCapture` (replay from `.pcap`/`.pcapng` files, with
   `--loop`). `include/spatial/libibverbs.hpp` has an alternate RDMA/ibverbs-based capture path.
   `src/ethernet.cpp` + `include/spatial/ethernet.hpp` parse raw Ethernet/IP/UDP/custom headers
   (`packet_formats.hpp` defines the on-wire `EthernetHeader`/`IPHeader`/`UDPHeader`/`CustomHeader`
   structs and the `LambdaPacketEntry`/`LambdaFinalPacketData` in-memory representations).

2. **Buffering / reassembly** (`ProcessorStateBase` / `ProcessorState<T, NR_INPUT_BUFFERS,
   RING_BUFFER_SIZE>` in `include/spatial/spatial.hpp`): a lock-free-ish double/triple-buffered
   ring buffer that accumulates incoming packets per channel/FPGA, tracks missing packets, handles
   FPGA-to-FPGA delay alignment (`fpga_delays_packet_aligned`/`fpga_delays_subpacket`), and once a
   buffer is full, hands a `FinalPacketData*` off to a `GPUPipeline` for processing
   (`release_buffer` is the hand-back once the GPU is done with that buffer).

3. **GPU pipeline** (`GPUPipeline` abstract interface in `include/spatial/pipeline_base.hpp`;
   concrete `Lambda*Pipeline<T>` implementations in `include/spatial/pipeline.hpp`, CUDA kernels in
   `src/spatial.cu`/`include/spatial/spatial.cuh`, tensor permutation helpers in
   `src/tensor.cpp`/`include/spatial/tensor.hpp`): converts/scales raw int8 samples to `__half`,
   correlates via `libtcc::Correlator` (Tensor Core Correlator, `extern/tcc`), beamforms via
   `ccglib` GEMM (`extern/ccglib`), computes eigendecompositions with cuSOLVER for adaptive
   spatial filtering / RFI subtraction, and runs FFTs via cuFFT for spectra/bandpass output. Key
   variants: `LambdaGPUPipeline` (baseline correlate+beamform), `LambdaAntennaSpectraPipeline`,
   `LambdaBeamformedSpectraPipeline`, `LambdaAdaptiveBeamformedSpectraPipeline` (RFI mitigation via
   eigenvector projection), `LambdaCorrBeamOnlyGPUPipeline`, `LambdaProjectionPipeline`,
   `LambdaPulsarFoldPipeline` (folding for pulsar timing).

4. **Output / writers** (`Output` interface + `SingleHostMemoryOutput<T>` /
   `BufferedOutput<T>` in `include/spatial/output.hpp`; concrete `Writer<T>` subclasses in
   `include/spatial/writers.hpp`): double-buffered host-memory landing zones that the GPU DMAs
   into asynchronously (`register_*_block` / `get_*_landing_pointer` /
   `register_*_transfer_complete`), drained by background writer threads to
   `HDF5BeamWriter`/`HDF5VisibilitiesWriter`/`HDF5FFTWriter`/`HDF5ProjectionEigenWriter` (HighFive),
   `PSRDADABeamWriter` (PSRDADA ring buffers), `RedisEigendataWriter`/`RedisBeamFFTWriter`
   (redis-plus-plus, for live monitoring), and CasaCore/CFITSIO-based outputs.

`include/spatial/common.hpp` is shared app glue: `parse_common_args` (the argparse setup every
binary uses — config/gains/PCAP/interface/etc. flags → `CommonArgs`), `setup_logger` (async spdlog
→ `app.log`), `get_gains_structure` (loads per-channel/pol/antenna calibration gains from JSON),
and `AntennaMapRegistry` (hard-coded FPGA-stream → physical-antenna-ID maps for the LAMBDA array).
Logging/error-check macros (`INFO_LOG`, `DEBUG_LOG`, `CUDA_CHECK`, `CUFFT_CHECK`, `CUBLAS_CHECK`,
`CUSOLVER_CHECK`) live in `include/spatial/logging.hpp`.

`src/` builds everything into a single `spatial` static library that all apps and tests link
against (see `src/CMakeLists.txt`). Note `src/packet_formats.cpp` and `src/pipeline.cpp` are
currently empty stubs, and `include/spatial/adaptive_pipelines.hpp` is an empty placeholder header.

### Apps (`apps/`, see `apps/CMakeLists.txt`)

All link `gpu_app_common` and share `parse_common_args`/`CommonArgs` from `common.hpp`. Each
hard-codes its own `LambdaConfig` instantiation in `main()`:

- `observe` — main live/PCAP-replay observing pipeline (correlate + beamform + dump visibilities).
- `beamformed_bandpass` — beamformed spectra/bandpass output.
- `adaptive_beamformed_bandpass` — adaptive (eigen-projection-based RFI-mitigated) beamforming.
- `get_projection_matrix` — computes/dumps the eigenvector projection matrix used for adaptive
  filtering (consumed later via HDF5 by `ProjectionWeightApplicator` in `beamformed-bandpass.cu`).
- `fft_antenna_spectra` — per-antenna spectra via FFT.
- `pulsar_fold` — folds beamformed data at a pulsar period.
- `gpu_benchmark` — GPU kernel/pipeline benchmarking harness.
- `udp_sender` — replays a PCAP file as a live UDP stream (for testing the live-capture path).

### Domain gotchas (from prior debugging — keep these in mind when touching GPU/tensor code)

- **TCC**: data points per block = `128 * COMPLEX / (N_BITS_PER_COMPONENT * COMPLEX)`, e.g. 16
  time points/block for 8-bit components. `NR_RECEIVERS` must be a multiple of 32 (zero-pad if
  fewer). `NR_POLARIZATIONS` must be 2 (zero-pad otherwise).
- **ccglib**: the second GEMM operand must be column-major only in its rows/columns sections — the
  Batch/Complex sections are *not* column-major.
- **cuTensor**: assumes column-major modes; `tensor.cpp` does a `std::reverse` on modes/extents so
  the rest of the code can reason in row-major terms.
- **gpu-filter's FIR kernel looks *forward*, not backward**: `readInputAndDoFIRfiltering`
  (`FilterAndCorrect.cu`) computes output tick `T` from raw input ticks `[T, T+NR_TAPS-1]` — the
  extra `NR_TAPS-1` samples its `InputType` reserves beyond the "real" sample count are **trailing
  look-ahead** for the last output block, not **leading history** before the first. It's an easy
  convention to get backwards by analogy with a causal FIR filter (which *would* want leading
  history) — `FineChannelizer`'s `reorder_to_filter_input` (`spatial.cuh`) writes real samples
  starting at buffer offset 0 and zero-pads the trailing region accordingly; get this backwards and
  every fine-channelized output is silently computed from zero/mismatched input, not caught by
  `FineChannelizerTests`' finite/autocorrelation invariant checks (both are trivially satisfied by
  degenerate near-zero output).
- **New cuTensor mode labels can perturb permutations that never reference them**: adding a mode
  label to one tensor (e.g. widening a `'c'`-labeled axis to a separate `'C'` for the raw
  pre-channelization axis) can change the numeric output of a *different*, unrelated permutation in
  the same `CutensorSetup` even when that permutation's own mode vector wasn't edited and the
  extents involved are numerically identical — observed once as a real regression
  (`LambdaAdaptiveBeamformedSpectraPipeline`'s `modePacketAligned`, changed from `'c'` to `'C'`
  alongside the mode actually needed by new code, broke an existing disabled-path test). Only
  change a mode label for a tensor that the new code path actually reads/writes via cuTensor; leave
  labels on tensors used solely by old/disabled-path-only permutations untouched, even if they're
  nominally shaped the same way.
- **`LambdaCorrBeamOnlyGPUPipeline` is the one pipeline where `NR_FINE_CHANNELS == 1` and `> 1` are
  genuinely different code, not the same path at a different width**: every other `Lambda*Pipeline`
  reuses the same delay/reorder/channelize/correlate/beamform sequence at both widths (the
  `if constexpr` branches differ only in whether the channelizer stage runs). This one's disabled
  path is a fused single-kernel fast path (`packet_to_corr_input`/`packet_to_col_maj_cons`, no
  delay/reorder stage at all — it's a benchmark harness measuring that steady-state overhead) with
  no seam for a channelizer, so the `NR_FINE_CHANNELS > 1` branch reconstructs the full
  delay-bridge-through-beamform sequence from scratch instead. Don't assume the two branches are
  numerically equivalent by construction the way they are everywhere else.

## Python tooling (`scripts/`, `ui/`)

- `scripts/` contains marimo notebooks (`correlate-pcap.py`, `read_visibilities.py`,
  `explore_metrics.ipynb`) for visualizing captured PCAP/HDF5 data — run with
  `source .venv/bin/activate && marimo edit` (see `scripts/README.md` for SSH port-forwarding to
  view the marimo UI remotely), `create_pcap.py`/`udp_sender` for synthesizing test packet
  captures, and `mlflow_save_benchmarks*.py` for logging benchmark runs to MLflow.
- `ui/ui.py` is a Textual TUI (`CMakeBuilder`) that exposes the `NR_OBSERVING_*`/`NUMBER_BEAMS`
  CMake cache variables as a form and drives `cmake`/`cmake --build` for you.

## Documentation

Maintain living architecture/data-flow documentation in `docs/`, and treat it as part of the
change, not an afterthought — any change that alters a pipeline's stages, buffer shapes/layouts,
or the relationships between components (`GPUPipeline` variants, `ProcessorState`, `Output`
writers, external libraries like `tcc`/`ccglib`/`gpu-filter`) should update the relevant doc in the
same commit.

- Prefer diagrams over prose for data flow, pipeline stage sequencing, buffer/tensor shape
  transformations, and component dependencies — a reader should be able to see the shape of the
  system before reading any code. Use **Mermaid** diagrams embedded directly in markdown
  (` ```mermaid ` code fences) as the default: they render as diagrams on GitHub/GitLab and in most
  markdown viewers/editors, and — unlike static image files — are plain text, so they stay
  accurate as the code changes and are easy for an agent (or a human) to update in a normal diff
  rather than regenerating and re-exporting a binary image. Reach for an actual image file (SVG/PNG
  under `docs/`) only when a diagram genuinely can't be expressed as Mermaid (e.g. a real hardware
  photo, an annotated screenshot, a complex physical layout).
- `docs/architecture.md`: top-level pipeline data-flow diagram — packet capture →
  `ProcessorState` buffering → `GPUPipeline` (correlate/beamform/eigendecompose) → `Output`
  writers — plus one diagram per `Lambda*Pipeline` variant showing its own stage sequence and where
  it diverges from the others.
- For any new multi-stage subsystem (e.g. a new pipeline variant, a new channelization/filtering
  stage), add or extend a diagram showing its buffer shapes at each stage boundary, not just prose
  describing them — this codebase leans heavily on compile-time array shapes
  (`LambdaConfig`-derived types) threaded through many reshape/permutation kernels, which is exactly
  the kind of thing that's much clearer as a diagram than as a paragraph.
- Keep doc updates scoped to what actually changed — don't regenerate an entire diagram set for an
  unrelated change, but do update the specific diagram(s) that describe whatever you touched.

## Notes

- Several large generated/data artifacts (`.hdf5`, `.pcap`/`.pcapng`, `.ncu-rep`, `.nsys-rep`,
  `dump.rdb`, `*.so`, etc.) live in the repo root and `scripts/` for benchmarking/profiling and
  analysis — they're build outputs or captured data, not source.
- `extern/` (git submodules: tcc, ccglib, cudawrappers, libpcap, spdlog, xtensor/xtl, argparse,
  highfive, googletest, rdma-core) is gitignored; the build `FetchContent_Declare`s each from its
  local `SOURCE_DIR`.
