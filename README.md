
# LAMBDA GPU Signal Processing Pipeline

Real-time radio-astronomy signal processing pipeline for the LAMBDA instrument. Ingests UDP/PCAP
packet streams from FPGA-based receivers, correlates and beamforms on the GPU via Tensor Core
Correlator and ccglib, applies adaptive spatial filtering / RFI mitigation via eigendecomposition,
and writes results to HDF5/PSRDADA/FITS/Redis.

See `CLAUDE.md` for the full architecture walkthrough, domain gotchas, and toolchain notes.

## Building

### In the Docker container (standard development environment)

```bash
mkdir build_test && cd build_test

# CUTENSOR_ROOT: /opt/conda has the flat include/lib layout find_cutensor.cmake expects
CUTENSOR_ROOT=/opt/conda cmake -DBUILD_TESTING=ON ..
cmake --build . -- -j$(nproc)
```

If you see link errors (`arc4random@GLIBC_2.36`, `strlcpy@GLIBC_2.38`, etc.), the conda g++
is resolving against its bundled older glibc. Add the sysroot flags:

```bash
CUTENSOR_ROOT=/opt/conda cmake -DBUILD_TESTING=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--sysroot=/ -L/usr/lib/x86_64-linux-gnu -Wl,-rpath-link,/usr/lib/x86_64-linux-gnu" \
  ..
```

### On OzStar

```bash
module load cuda/12.6.0 gcc/13.3.0 cmake/3.29.3
mkdir build && cd build
cmake -DBUILD_TESTING=ON ..
cmake --build . -- -j$(nproc)
```

## Running Tests

```bash
cd build_test

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
