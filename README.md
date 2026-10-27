
# LAMBDA GPU Signal Processing Pipeline

[![Build](https://github.com/jdgsmallwood/cuda-spatial-filtering/actions/workflows/build.yml/badge.svg)](https://github.com/jdgsmallwood/cuda-spatial-filtering/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

Real-time CUDA C++ signal processing pipeline for the LAMBDA radio-astronomy instrument. Ingests
UDP/PCAP packet streams from FPGA-based receivers, correlates and beamforms on the GPU via Tensor
Core Correlator and ccglib, applies adaptive spatial filtering and RFI mitigation via
eigendecomposition, and writes results to HDF5/PSRDADA/FITS/Redis.

## Statement of Need

Low-frequency radio telescopes face an increasingly hostile radio-frequency interference (RFI)
environment, requiring real-time adaptive spatial filtering that is computationally too demanding
for CPU-only approaches. Existing GPU correlator frameworks (e.g., xGPU) target standard
interferometric arrays and do not expose the beamforming, eigendecomposition-based RFI projection,
or the bespoke LAMBDA FPGA wire-format parsing required for this instrument. This pipeline
provides a GPU path from FPGA packets to calibrated beams and visibilities, with optional RFI
mitigation. In separate component benchmarks, CPU packet reassembly processed about 4.6 million
packets/s, while an 8-channel, 4-FPGA GPU correlate-and-beamform benchmark achieved 531% of its
real-time block rate. These figures do not measure combined pipeline throughput. In a 900-second
live, 40-channel, 4-FPGA pulsar-fold run, the pipeline received 2,083,384,981 packets (about
2.31 million packets/s) and sustained about 56.5 blocks/s.

## Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| CUDA | ≥12.6 | cuSOLVER, cuFFT, cuBLAS, cuTENSOR, NVRTC required |
| cuTENSOR | ≥2.2 | `CUTENSOR_ROOT` must point at a flat `include/lib` layout |
| HDF5 | ≥1.10 | With C bindings |
| CFITSIO | ≥4.0 | |
| CasaCore | ≥3.5 | |
| PSRDADA | any | `$PSRHOME` must point at install prefix |
| CMake | ≥3.15 | |

Dependencies in `extern/` (TCC, ccglib, cudawrappers, spdlog, xtensor, argparse, HighFive,
googletest, redis-plus-plus) are git submodules — run
`git submodule update --init --recursive` if they are missing.

## Building

### In the Docker container (standard development environment)

```bash
mkdir -p build

# CUTENSOR_ROOT: /opt/conda has the flat include/lib layout find_cutensor.cmake expects
CUTENSOR_ROOT=/opt/conda cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -- -j$(nproc)
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
export CUDA_HOME=/opt/conda/targets/x86_64-linux

ctest                                      # run everything
ctest -R ProcessorTests                    # filter by test suite (regex)
./tests/CorrBeamPipelineTests --gtest_filter='*ExactValues*'  # run gtest binary directly
```

## Test Suites

| Binary | Source | What it covers |
|---|---|---|
| `CUDASpatialFilteringCPUTests` | `test_packet_formats.cpp` | Wire-format packet parsing: Ethernet/IP/UDP/Custom header extraction, short-packet handling, multi-position sample layout, FPGA-ID parsing modes |
| `ProcessorTests` | `test_processor.cu` | Ring-buffer reassembly: single-packet ingestion, buffer fill/flush, multi-FPGA placement and delay alignment, frequency-range filtering, missing-packet accounting |
| `PipelineTests` | `test_pipeline.cu` | `LambdaGPUPipeline` GPU kernels: exact-value golden checks for beam/visibility output from `Ex1`, polarization/channel/scale blanking |
| `CorrBeamPipelineTests` | `test_corr_beam_pipeline.cu` | `LambdaCorrBeamOnlyGPUPipeline`: exact-value beam and visibility checks, physical invariants (autocorrelation non-negativity, Hermitian covariance), zero-weight / zero-input edge cases, multi-channel independence, multi-packet accumulation |
| `PipelineHarnessSelfTest` | `support/test_harness_selftest.cu` | Validates the shared pipeline test harness reproduces `Ex1` exact values and passes physical invariants end-to-end through the real `ProcessorState → Pipeline → Output` seam |
| `PulsarFoldTests` | `test_pulsar_fold_pipeline.cu` | `LambdaPulsarFoldPipeline`: beams are non-zero with non-zero weights, zero-weights produce zero output, tracking weights are applied |
| `WriterTests` | `test_writers.cpp` | HDF5 beam/visibility/FFT writer round-trips |
| `CUDASpatialFilteringTests` | `test_spatial.cpp` | Legacy correlation experiments (mostly disabled) |

See `tests/TESTING.md` for the design principles behind the shared harness in `tests/support/`.

## Domain Notes

**TCC (Tensor Core Correlator)**
- Data points per block = 128 / NR_BITS_PER_COMPONENT (e.g. 16 time points/block for 8-bit).
- `NR_RECEIVERS` must be a multiple of 32; zero-pad if fewer.
- `NR_POLARIZATIONS` must be 2; zero-pad if needed.

**ccglib**
- The second GEMM operand must be column-major for its rows/columns sections only — Batch/Complex
  sections are not column-major.

**cuTENSOR**
- Assumes column-major modes. `tensor.cpp` does `std::reverse` on modes/extents so the rest of
  the code can reason in row-major terms.
