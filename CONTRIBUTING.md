# Contributing

Thank you for your interest in contributing to the LAMBDA GPU Signal Processing Pipeline.

## Setting up a development environment

### Docker (recommended)

The Docker image encapsulates all dependencies. Build it from the repo root:

```bash
docker build -t lambda-gpu-pipeline .
docker run --gpus all -it --rm -v "$PWD":/workspace lambda-gpu-pipeline
```

Inside the container, configure and build:

```bash
cd /workspace
mkdir build_test && cd build_test
CUTENSOR_ROOT=/opt/conda cmake -DBUILD_TESTING=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--sysroot=/ -L/usr/lib/x86_64-linux-gnu -Wl,-rpath-link,/usr/lib/x86_64-linux-gnu" \
  ..
cmake --build . -- -j$(nproc)
```

If you see link errors mentioning `arc4random@GLIBC_2.36` or similar, the `DCMAKE_EXE_LINKER_FLAGS`
line above (already included) is the fix — the conda `g++` resolves against an older bundled glibc.

### OzStar HPC cluster

```bash
module load cuda/12.6.0 gcc/13.3.0 cmake/3.29.3
mkdir build && cd build
cmake -DBUILD_TESTING=ON ..
cmake --build . -- -j$(nproc)
```

### Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| CUDA | ≥12.6 | cuSOLVER, cuFFT, cuBLAS, cuTENSOR, NVRTC required |
| cuTENSOR | ≥2.2 | Set `CUTENSOR_ROOT` to the flat `include/lib` layout |
| HDF5 | ≥1.10 | With C bindings |
| CFITSIO | ≥4.0 | |
| CasaCore | ≥3.5 | |
| PSRDADA | any | `$PSRHOME` must point at install prefix |
| CMake | ≥3.15 | |

Dependencies in `extern/` (TCC, ccglib, cudawrappers, spdlog, xtensor, argparse, HighFive,
googletest, redis-plus-plus) are git submodules — run
`git submodule update --init --recursive` if they are missing.

## Running the tests

```bash
cd build_test
export CUDA_HOME=/opt/conda/targets/x86_64-linux   # needed by NVRTC at runtime

ctest                                               # run all suites
ctest -R ProcessorTests                             # run one suite by name
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
