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

Test binaries (see `tests/CMakeLists.txt`): `CUDASpatialFilteringTests` (test_spatial.cpp, mostly
disabled/commented-out correlation experiments), `PipelineTests` (test_pipeline.cu, GPU pipeline
kernels), `CUDASpatialFilteringCPUTests` (test_packet_formats.cpp, CPU-only packet parsing),
`WriterTests` (test_writers.cpp, HDF5 writers), `ProcessorTests` (test_processor.cu, the packet
ring-buffer/`ProcessorState` machinery), `PipelineHarnessSelfTest` (support/test_harness_selftest.cu,
proves out the shared pipeline-test harness in `tests/support/` — see `tests/TESTING.md`), plus
`test_beamforming.cpp` for beamforming math.

Build options that shape the generated code (set with `-D...`, see top-level `CMakeLists.txt`):
`NR_OBSERVING_BUFFERS`, `NR_OBSERVING_FPGA_SOURCES`, `NR_OBSERVING_CHANNELS`,
`NR_OBSERVING_RECEIVERS_PER_PACKET`, `NR_OBSERVING_PADDED_RECEIVERS`,
`NR_OBSERVING_PACKETS_FOR_CORRELATION`, `NR_OBSERVING_CORRELATION_BLOCKS_TO_INTEGRATE`,
`NUMBER_BEAMS`. These become `add_compile_definitions` and feed directly into the `LambdaConfig`
template instantiations in each app's `main()`. They only apply to the `apps` subdirectory build
(skipped when `BUILD_TESTING=ON`).

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

## Python tooling (`scripts/`, `ui/`)

- `scripts/` contains marimo notebooks (`correlate-pcap.py`, `read_visibilities.py`,
  `explore_metrics.ipynb`) for visualizing captured PCAP/HDF5 data — run with
  `source .venv/bin/activate && marimo edit` (see `scripts/README.md` for SSH port-forwarding to
  view the marimo UI remotely), `create_pcap.py`/`udp_sender` for synthesizing test packet
  captures, and `mlflow_save_benchmarks*.py` for logging benchmark runs to MLflow.
- `ui/ui.py` is a Textual TUI (`CMakeBuilder`) that exposes the `NR_OBSERVING_*`/`NUMBER_BEAMS`
  CMake cache variables as a form and drives `cmake`/`cmake --build` for you.

## Notes

- Several large generated/data artifacts (`.hdf5`, `.pcap`/`.pcapng`, `.ncu-rep`, `.nsys-rep`,
  `dump.rdb`, `*.so`, etc.) live in the repo root and `scripts/` for benchmarking/profiling and
  analysis — they're build outputs or captured data, not source.
- `extern/` (git submodules: tcc, ccglib, cudawrappers, libpcap, spdlog, xtensor/xtl, argparse,
  highfive, googletest, rdma-core) is gitignored; the build `FetchContent_Declare`s each from its
  local `SOURCE_DIR`.
