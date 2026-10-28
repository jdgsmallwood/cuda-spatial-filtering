# Contributing

Thank you for your interest in contributing to Starspin.

## Setting up a development environment

### Docker (recommended)

The Docker image includes the core Starspin dependencies. Build it from the repo root:

```bash
docker build -t starspin-dev .
docker run --gpus all -it --rm -v "$PWD":/workspace starspin-dev
```

Follow the [reviewer guide](docs/reviewer-guide.md) to configure the core targets,
run the GPU tests, and check the offline replay. Use an explicit GPU architecture
and a fresh build directory. The container does not include optional DSPSR or
PSRCHIVE tooling.

### OzStar HPC cluster

```bash
module load cuda/12.6.0 gcc/13.3.0 cmake/3.29.3
mkdir build && cd build
cmake -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=80 ..  # example for an A100
cmake --build . -- -j$(nproc)
```

### Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| CUDA | ≥12.6 | cuSOLVER, cuFFT, cuBLAS, cuTENSOR, NVRTC required |
| cuTENSOR | ≥2.2 | Set `CUTENSOR_ROOT` for a non-standard installation |
| HDF5 | ≥1.10 | With C bindings |
| CFITSIO | ≥4.0 | |
| CasaCore | ≥3.4 | |
| PSRDADA | pinned in Dockerfile | `$PSRHOME` points at the install prefix |
| Boost, FFTW3, hiredis | distribution development packages | Required by gpu-filter and Redis output dependencies |
| CMake | ≥3.28 | |

Dependencies in `extern/` (TCC, ccglib, cudawrappers, spdlog, xtensor, argparse, HighFive,
googletest) are git submodules — run
`git submodule update --init --recursive` if they are missing.

## Running the tests

```bash
cd build
export CUDA_HOME=/usr/local/cuda   # needed by NVRTC at runtime

ctest --output-on-failure                           # after building all test targets
./tests/ProcessorTests                             # run the processor suite
./tests/CorrBeamPipelineTests --gtest_filter='*ExactValues*'  # run a single test
```

GPU tests require a physical NVIDIA GPU. See `tests/TESTING.md` for the full testing strategy and
how to add new tests using the shared harness in `tests/support/`.

## Code style

- C++ follows the conventions already present in the codebase (clang-format configuration is
  in progress; match the style of nearby code when in doubt).
- CUDA kernels live in `src/spatial.cu` / `include/spatial/spatial.cuh`. New kernels should have a
  corresponding test in `tests/` using the shared pipeline harness.
- Template parameters follow the `LambdaConfig<T>` convention — prefer extending the existing
  template rather than adding free functions.
- No comments that restate what the code already says. Comments should explain *why* (hidden
  constraints, non-obvious invariants, CUDA-specific gotchas).

## Opening issues and pull requests

- Use the GitHub issue tracker for bug reports and feature requests.
- For bug reports, include the CUDA device and driver version, the CMake configuration used, and a
  minimal reproducer.
- Pull requests are reviewed on GitHub. Please ensure tests pass before requesting review.
- All contributions are made under the MIT license.

## Support and governance

This project is maintained by Justin Smallwood. Questions can be directed via GitHub issues.
Community contributions are welcome; the maintainer aims to respond to issues and pull requests
within two weeks.

## Verification and releases

Install `requirements-test.txt` and run `python3 -m pytest -q -ra tests` for the
CPU reference tests. Missing `lambda_commissioning` skips its integration tests;
a skipped integration is not evidence of correctness. GitHub Actions checks
these tests and citation metadata. GPU verification is documented separately.

Use the [JOSS readiness checklist](docs/joss-readiness.md) for release evidence,
authorship confirmation, archive preparation, and submission gates.
