# Starspin reviewer guide

This guide exercises the correlator used for LAMBDA commissioning without a
network interface or telescope. The small generated CODIF fixture passes through
the real application and HDF5 writer, then an independent NumPy reference checks
its visibilities. See [readiness](joss-readiness.md) for current validation results
and outstanding release gates.

## Source and environment

```bash
git clone https://github.com/jdgsmallwood/cuda-spatial-filtering.git
cd cuda-spatial-filtering
git submodule update --init --recursive
```

**Current release blocker:** the recorded `extern/tcc` commit
`d71f1b4d0f1dc03f02bedde738fc9b80818a42fd` is unavailable from its configured public
remote. Do not substitute an arbitrary upstream version. Local preparation is
using the preserved `36e5edc3bc49772670edb5451f7b397342d26602` revision at the
maintainer's direction while the TCC publication issue is resolved. This does
not yet establish that an independent recursive clone can build.

The core environment is Linux x86-64, C++17, CUDA 12.8, CMake 3.28 or later,
cuTENSOR 2.2, HDF5, CFITSIO, casacore, PSRDADA, Boost, FFTW3, and hiredis. The Dockerfile
installs these dependencies and the Python test requirements. Git submodules
and CMake FetchContent provide the remaining libraries. Internet access is
needed during image construction and initial CMake configuration. The host
needs a compatible NVIDIA driver and NVIDIA Container Toolkit.

```bash
docker build -t starspin-dev .
docker run --rm --gpus all -it -v "$PWD":/workspace starspin-dev
```

The preparation host required `--network=host` on `docker build` and on the
initial configure container because dependency downloads stalled over its
Docker bridge. Use that Linux networking option if the same problem occurs;
the replay and tests themselves can run with `--network none`.


Use a Tensor-Core-capable NVIDIA GPU (FP16 support begins at sm_70); the
preparation machine is an 8 GiB RTX 4060. This is a tested device, not a measured
minimum-memory guarantee. Use the GPU's numeric CUDA architecture (`89` for an RTX 4060). Build the small
review configuration first; production configurations have different memory
requirements. The core image does not include DSPSR or PSRCHIVE. Do not use a
build directory created in another environment.

```bash
cmake -S . -B build -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --parallel 2 --target \
  starspin_1_8 StarspinTests FineChannelizedStarspinTests \
  CUDASpatialFilteringCPUTests ProcessorTests WriterTests RunAuditTests
```

The full C++ build requires CUDA even for its CPU test targets. There is no
CPU-only implementation of the Starspin application. The Python reference tests
can run independently:

```bash
python3 -m pip install -r requirements-test.txt
python3 -m pytest -q -ra tests
```

Some sky-calibration integration tests need the separate `lambda_commissioning`
package and report explicit skips when it is absent. The GitHub Checks workflow
covers Python tests and citation metadata; it does not imply GPU tests passed.

## GPU checks and replay

```bash
export CUDA_HOME=/usr/local/cuda
./build/tests/StarspinTests
./build/tests/FineChannelizedStarspinTests
./build/tests/CUDASpatialFilteringCPUTests
./build/tests/ProcessorTests
./build/tests/WriterTests
./build/tests/RunAuditTests
python3 scripts/review_starspin.py --binary build/apps/starspin_1_8 \
  --output-dir build/reviewer-replay
```

Use a new output directory on each replay. The runner refuses to overwrite an
existing result. It generates zero delays and empty gain/eigenvalue inputs for
this **single-FPGA synthetic validation**, not for live observing. It loops the
85,848-byte fixture for three seconds, integrates four correlation blocks per
output, and saves the command, log, HDF5 file, audit sidecar, NumPy reference,
and `reference/comparison.json`.

Success requires `passed: true`, relative RMS error below `1e-3`, maximum
error normalized by reference autopower below `2e-3`, and zero missing packets
in each compared integration. The number of integrations is timing-dependent;
the reference uses the actual sequence range. It checks every complex output
and requires the conjugation and missing-scale negative controls to fail the
acceptance tolerance. This is a correctness test, not a real-time benchmark.

For a full GPU regression run, build the remaining test targets and run
`ctest --test-dir build --output-on-failure`. Do not interpret unbuilt tests or
skips as passes. `tests/TESTING.md` describes the synthetic packet harness.

## Live commissioning and performance

For an external coarse-channel correlation check, see
[Starspin–DiFX verification](difx-validation.md). This uses DiFX's `zerocorr`
utility and reports CPU-only checks separately from GPU comparison results.

Use an external [deployment profile](deployment.md) and the site's observing
procedure. Verify the FPGA boot state, tuning, matching delay calibration, and
delay closure before evaluating live data. Retain those records with the capture.
The loopback fixture's zero delays are only for synthetic tests. A local kernel
socket exercise is described in [kernel loopback](kernel-loopback.md); it does
not verify hardware capture or calibration.

## Colleague rehearsal

A colleague should repeat the source download, build, tests, and replay on a
fresh machine or environment, preserving commands and failures. Record the
commit, submodule revisions, image ID, GPU/driver, test counts, skipped tests,
and replay report. A maintainer's local run is not an independent colleague
rehearsal. File reproducible problems through the repository issue tracker.

## Optional monitoring demonstration

The [monitoring guide](../monitoring/README.md) includes a provisioned Redis/Grafana
stack, CPU-only synthetic dashboard checks, and numerical replay commands for
monitoring enabled, disabled and Redis unavailable. This optional service is
not required for the core reviewer workflow.
