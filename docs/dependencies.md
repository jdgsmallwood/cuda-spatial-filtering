# Starspin dependency provenance

Starspin's own source is MIT licensed. Dependencies retain their own licences;
the top-level licence does not relicense them. Preserve their licence files
and notices when distributing source, containers, or linked binaries.

| Dependency | Source or version selection |
| --- | --- |
| Tensor-Core Correlator | `extern/tcc` gitlink; current public availability blocker in `joss-readiness.md` |
| ccglib, cudawrappers, xtensor, xtl, argparse, HighFive, spdlog, libpcap, googletest | Exact gitlinks in `.gitmodules` and the superproject tree |
| gpu-filter | ASTRON, CMake revision `2026-09-03`; transitive dependencies recorded in its source |
| nlohmann/json | Release 3.12.0 fetched by CMake |
| redis-plus-plus | Release 1.3.11 fetched by CMake |
| hiredis | System development package in the core image |
| PSRDADA | SourceForge revision `151198ea3ec4a3a237f51e6217a5a2d7ff39194e` in Dockerfile |
| CUDA and cuTENSOR | NVIDIA CUDA 12.8.1 image and cuTENSOR 2.2 installer; NVIDIA licences apply |
| cuFFTDx/mathdx | gpu-filter's configured dependency; NVIDIA licence applies |
| HDF5, CFITSIO, casacore, Boost, FFTW3, libibverbs | Linux distribution packages in Dockerfile |
| Python verification libraries | `requirements-test.txt` |

The Dockerfile pins major upstream source inputs but distribution packages and
Python compatible-version ranges can change. Record the built image digest and
`python3 -m pip freeze` with release evidence. Record transitive CMake source
revisions as well as submodules. This environment does not include the optional
`lambda_commissioning` sky model, DSPSR, or PSRCHIVE.

Before distribution, check each dependency's exact revision and redistribution
terms. In particular, casacore/PSRDADA and copied build modules carry licences
separate from Starspin's MIT licence. `cmake/FindCasaCore.cmake` contains its own
GPL notice. Retain these notices and confirm obligations for the chosen archive
or binary distribution. No claim that a container is wholly MIT licensed is
made here.
