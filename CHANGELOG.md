# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Isolated kernel-UDP loopback guide and paced synthetic CODIF sender.
- Reproducible synthetic fixture with no commissioning samples or timestamps.
- JOSS paper draft and reviewer guide focused on the Starspin correlator used in LAMBDA commissioning.
- Offline CODIF replay runner using the existing independent NumPy visibility reference.
- Python/citation checks and paper-generation workflows.
- Output conventions, dependency provenance, and explicit submission/release gates.

### Fixed
- Deployment antenna maps and launcher settings now come from external site configuration; private operational records are no longer part of the public tree.
- Companion beam pipelines copy packet-arrival metadata before releasing the input buffer, with regression tests for all three affected paths.
- Core Docker environment no longer depends on an untracked conda environment file.
- Removed an unused C atomic header incompatible with GCC 11 C++ builds.
- Propagated the JSON dependency required by public headers to all consumers, including tests.
- Redis dependency tag, installed hiredis discovery, and minimum CMake requirement match the build configuration.
- Optional LAMBDA sky-model tests report missing integration dependencies explicitly.
- Citation metadata uses the supplied authors and no placeholder ORCID or unassigned paper DOI.


## 0.1.0 development milestone - 2026-08-12

Historical development milestone of the LAMBDA GPU Signal Processing Pipeline.
No corresponding public Git tag or GitHub release was present when checked on
15 October 2026.

### Added

**Core pipeline**
- Real-time UDP packet capture from FPGA-based receivers via kernel socket (`KernelSocketPacketCapture`) and PCAP replay (`PCAPPacketCapture`, `PCAPMultiFPGAPacketCapture`)
- Lock-free ring-buffer packet reassembly with per-channel/FPGA tracking, missing-packet accounting, and FPGA-to-FPGA delay alignment (`ProcessorState`)
- Ethernet/IP/UDP/custom header parsing for the LAMBDA wire format (`packet_formats.hpp`)

**GPU processing**
- Tensor Core Correlator (TCC) integration for GPU-accelerated visibility computation (`LambdaGPUPipeline`)
- Multi-beam beamforming via ccglib GEMM (`LambdaCorrBeamOnlyGPUPipeline`)
- Adaptive spatial filtering / RFI mitigation via cuSOLVER eigendecomposition (`LambdaAdaptiveBeamformedSpectraPipeline`)
- Per-antenna spectra via cuFFT (`LambdaAntennaSpectraPipeline`)
- Pulsar period folding (`LambdaPulsarFoldPipeline`)
- Eigenvector projection matrix computation (`LambdaProjectionPipeline`)

**Output writers**
- HDF5 beam, visibility, FFT, and projection-eigenvector writers (HighFive)
- PSRDADA ring-buffer output for real-time downstream processing
- Redis streams for live monitoring (`RedisEigendataWriter`, `RedisBeamFFTWriter`)
- FITS/CasaCore output

**Compile-time configuration**
- `LambdaConfig<...>` template parameterises the entire pipeline over number of channels, FPGA sources, receivers, packets per correlation, and beams at compile time
- CMake build options: `NR_OBSERVING_CHANNELS`, `NR_OBSERVING_FPGA_SOURCES`, `NR_OBSERVING_RECEIVERS_PER_PACKET`, `NR_OBSERVING_PADDED_RECEIVERS`, `NR_OBSERVING_PACKETS_FOR_CORRELATION`, `NR_OBSERVING_CORRELATION_BLOCKS_TO_INTEGRATE`, `NUMBER_BEAMS`

**Applications**
- `observe` — main correlate + beamform + dump-visibilities pipeline
- `beamformed_bandpass` — beamformed spectra/bandpass output
- `adaptive_beamformed_bandpass` — RFI-mitigated adaptive beamforming
- `get_projection_matrix` — eigenvector projection matrix computation
- `fft_antenna_spectra` — per-antenna spectra
- `pulsar_fold` — pulsar period folding
- `gpu_benchmark` — performance characterisation harness
- `udp_sender` — PCAP-to-UDP replay for live-capture-path testing

**Python/analysis tooling**
- marimo notebooks for PCAP and HDF5 data analysis (`scripts/`)
- Textual TUI for setting CMake cache variables and driving builds (`ui/`)
- Benchmark and profiling tooling with MLflow integration

**Testing**
- 15 test binaries covering packet parsing, GPU kernels, ring-buffer reassembly, beamforming math, pointing, writers, and pulsar folding
- Shared pipeline test harness (`tests/support/`) with synthetic packet generation and invariant-based assertions

[Unreleased]: https://github.com/jdgsmallwood/cuda-spatial-filtering/commits/main
