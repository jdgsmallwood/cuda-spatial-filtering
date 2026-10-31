
# Starspin

[![Checks](https://github.com/jdgsmallwood/cuda-spatial-filtering/actions/workflows/checks.yml/badge.svg)](https://github.com/jdgsmallwood/cuda-spatial-filtering/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

Starspin is a CUDA C++ correlator used for commissioning the **LAMBDA telescope**.
It ingests live FPGA packets or recorded PCAP files, aligns antenna voltages,
optionally applies fine channelization, and writes integrated full-polarization
visibilities and capture provenance to HDF5.

Start with the [reviewer guide](docs/reviewer-guide.md) for installation and a
small offline replay with an independent numerical check. See the
[output format](docs/starspin-output.md), [architecture](docs/architecture.md),
and [deployment configuration](docs/deployment.md) and [kernel loopback test](docs/kernel-loopback.md) for details.
The [submission checklist](docs/joss-readiness.md) records remaining release gates.

## Statement of need

Commissioning a radio telescope requires tracing unexpected correlations back to
packet loss, stream mapping, delay alignment, channelization, or the instrument.
Starspin connects LAMBDA's packet formats and calibration inputs to GPU correlation,
retaining configuration and data-quality metadata alongside the visibilities.
Recorded-packet replay allows the same processing path to be checked without an
operating telescope. Starspin uses ASTRON's Tensor-Core Correlator and gpu-filter
rather than implementing replacement correlation and PFB libraries.

This repository also contains companion beamforming, adaptive spatial-filtering,
and pulsar applications. Those applications share infrastructure but their outputs
and performance are distinct from Starspin's correlator-only workflow. GPU kernel
benchmarks and host-only ingest rates must not be interpreted as end-to-end
Starspin throughput.

## Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| CUDA | ≥12.6 | cuSOLVER, cuFFT, cuBLAS, cuTENSOR, NVRTC required |
| cuTENSOR | ≥2.2 | Set `CUTENSOR_ROOT` for a non-standard installation |
| HDF5 | ≥1.10 | With C bindings |
| CFITSIO | ≥4.0 | |
| CasaCore | ≥3.4 | |
| PSRDADA | pinned in Dockerfile | `$PSRHOME` points at the install prefix |
| Boost, FFTW3, hiredis | distribution development packages | Required by gpu-filter and Redis output dependencies |
| CMake | ≥3.28 | FetchContent and cmake_path support |

Dependencies in `extern/` (TCC, ccglib, cudawrappers, spdlog, xtensor, argparse, HighFive,
googletest, xtl, libpcap) are git submodules — run
`git submodule update --init --recursive` if they are missing.

## Building

### In the Docker container

Build and launch the core environment (requires NVIDIA Container Toolkit):

```bash
docker build -t starspin-dev .
docker run --rm --gpus all -it -v "$PWD":/workspace starspin-dev
```

The container includes the core correlator dependencies, not DSPSR or PSRCHIVE.
Use an explicit CUDA architecture; `89` below is for an RTX 4060.
See the reviewer guide for the current clean-clone status.

```bash
mkdir -p build

cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target starspin_1_8 StarspinTests --parallel 2
```

If you see link errors (`arc4random@GLIBC_2.36`, `strlcpy@GLIBC_2.38`, etc.), the conda g++
is resolving against its bundled older glibc. Add the sysroot flags:

```bash
CUTENSOR_ROOT=/opt/conda cmake -DBUILD_TESTING=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--sysroot=/ -L/usr/lib/x86_64-linux-gnu -Wl,-rpath-link,/usr/lib/x86_64-linux-gnu" \
  -S . -B build
```

### On OzStar

```bash
module load cuda/12.6.0 gcc/13.3.0 cmake/3.29.3
mkdir -p build
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -- -j$(nproc)
```

### Installing binaries

Build and install all application binaries (including variants and benchmarks):

```bash
export STARSPIN_DIR=/path/to/starspin
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$STARSPIN_DIR"
cmake --build build --parallel 2
cmake --install build
```

Binaries are installed into `$STARSPIN_DIR/bin`. On a fresh configuration,
`STARSPIN_DIR` supplies the default install prefix; if unset or empty, the
project root is used. The explicit prefix above also updates existing build
caches. Override the destination with `cmake --install build --prefix /other/path`
(which installs into `/other/path/bin`). Test executables are not installed.

To install only the application binaries built for an observing run, use the
`observing` component from the `apps` build directory. Unbuilt app variants
are skipped, and this avoids unrelated dependency install rules:

```bash
cmake --build build --target starspin_4_40 pulsar_fold_intensity_4_40 starspin_4_8_coarse --parallel 4
cmake --install build/apps --component observing --prefix "$STARSPIN_DIR"
```

For ibverbs capture, grant `cap_net_raw=ep` to each installed binary after
installing it. Reinstalling a binary removes its previous file capability.

### Eight-channel Starspin without fine channelization

Build `starspin_1_8_coarse`, `starspin_2_8_coarse`, `starspin_3_8_coarse`, or
`starspin_4_8_coarse` for the required FPGA count. For four FPGAs:

```bash
cmake -S . -B build
cmake --build build --target starspin_4_8_coarse --parallel 4
```

Run `build/apps/starspin_4_8_coarse` with the usual capture, delay, mapping and
packet-format options. These targets correlate the eight incoming FPGA channels
directly, with no fine PFB, FFT or fine-bin trimming. Delay correction, scaling,
gains and stream reordering still apply. Internally, `NR_FINE_CHANNELS=1` means
channelization is disabled; the HDF5 output has eight channels.

### LAMBDA CODIF input

Use `--packet-format codif` for CODIF, `--packet-format lambda` for the original
packets, or `auto` (the default). CODIF supports kernel capture, ordinary ibverbs
CPU-ring capture, and PCAP replay; GPUDirect still requires legacy packets.

The October 2026 LAMBDA standard uses metadata ID `0x03BB`, Group ID equal to
the source IP's third octet, Secondary ID zero, and Station ID `At`. FPGA routing
continues to use the source IP's third octet. Replicated PCAP replay updates both
the source IP and Group ID for the new layout. The original simulation capture's
metadata ID zero and placeholder IDs remain supported; other metadata IDs and
Thread IDs outside 0–511 are rejected. Station/Secondary IDs and reserved bytes
are not used for routing or validation.

Samples and scaling are unchanged: 64 time samples of 20 complex signed 8-bit ADC
values, with per-ADC scale multipliers `2^min(exponent, 8)` from header bytes 46–55
(low nibble first). Frame counters wrap after 390,625 packets per 27 seconds.
The header's sample-period field accepts both LAMBDA's documented frame count
(390,625) and the corresponding sample count (25,000,000).

## Running Tests

```bash
cd build

# CUDA_HOME is needed at runtime by any test that constructs a LambdaGPUPipeline
# (it JIT-compiles TCCorrelator.cu via NVRTC and needs the CUDA toolkit headers)
export CUDA_HOME=/usr/local/cuda

ctest --output-on-failure                   # after building all test targets
./tests/StarspinTests                      # run the core correlator suite
./tests/CorrBeamPipelineTests --gtest_filter='*ExactValues*'  # run gtest binary directly
```

Python checks (no GPU required):

```bash
python3 -m pip install -r requirements-test.txt
python3 -m pytest -q -ra tests
```

Tests needing the separate `lambda_commissioning` package explicitly skip when
it is absent. The Checks badge covers Python and citation checks, not GPU tests.

## Test Suites

| Binary | Source | What it covers |
|---|---|---|
| `StarspinTests` | `test_starspin_pipeline.cu` | Coarse correlation values, channel independence, zero input, and invariants |
| `FineChannelizedStarspinTests` | `test_fine_channelized_starspin.cu` | Fine-channel correlation, delay signs, receiver mapping and compact receiver configurations |
| `CUDASpatialFilteringCPUTests` | `test_packet_formats.cpp` | Wire-format packet parsing: Ethernet/IP/UDP/Custom header extraction, short-packet handling, multi-position sample layout, FPGA-ID parsing modes |
| `ProcessorTests` | `test_processor.cu` | Ring-buffer reassembly: single-packet ingestion, buffer fill/flush, multi-FPGA placement and delay alignment, frequency-range filtering, missing-packet accounting |
| `PipelineTests` | `test_pipeline.cu` | `LambdaGPUPipeline` GPU kernels: exact-value golden checks for beam/visibility output from `Ex1`, polarization/channel/scale blanking |
| `CorrBeamPipelineTests` | `test_corr_beam_pipeline.cu` | `LambdaCorrBeamOnlyGPUPipeline`: exact-value beam and visibility checks, physical invariants (autocorrelation non-negativity, Hermitian covariance), zero-weight / zero-input edge cases, multi-channel independence, multi-packet accumulation |
| `PipelineHarnessSelfTest` | `support/test_harness_selftest.cu` | Validates the shared pipeline test harness reproduces `Ex1` exact values and passes physical invariants end-to-end through the real `ProcessorState → Pipeline → Output` seam |
| `PulsarFoldTests` | `test_pulsar_fold.cu` | `LambdaPulsarFoldPipeline`: beams are non-zero with non-zero weights, zero-weights produce zero output, tracking weights are applied |
| `WriterTests` | `test_writers.cpp` | HDF5 beam/visibility/FFT writer round-trips |
| `CUDASpatialFilteringTests` | `test_spatial.cpp` | Legacy correlation experiments (mostly disabled) |

See `tests/TESTING.md` for the design principles behind the shared harness in `tests/support/`.

## Domain Notes

**TCC (Tensor Core Correlator)**
- Data points per block = 128 / NR_BITS_PER_COMPONENT (e.g. 16 time points/block for 8-bit).
- Receiver padding depends on the target. Fine-channelized Starspin supports partial
  32-receiver tiles, including the 36-receiver production configuration.
- `NR_POLARIZATIONS` must be 2; zero-pad if needed.

**ccglib**
- The second GEMM operand must be column-major for its rows/columns sections only — Batch/Complex
  sections are not column-major.

**cuTENSOR**
- Assumes column-major modes. `tensor.cpp` does `std::reverse` on modes/extents so the rest of
  the code can reason in row-major terms.

### Optional live monitoring

[Redis/Grafana monitoring](monitoring/README.md) provides sampled visibility spectra,
packet statistics, per-core CPU and server/process memory. Enable it with
`--monitor`; unavailable Redis never blocks correlation. The guide includes a
localhost-only Compose stack and a synthetic replay demonstration.
