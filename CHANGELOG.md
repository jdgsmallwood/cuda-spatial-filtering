# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-08-12

Initial public release of the LAMBDA GPU Signal Processing Pipeline.

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

[Unreleased]: https://github.com/jdgsmallwood/cuda-spatial-filtering/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jdgsmallwood/cuda-spatial-filtering/releases/tag/v0.1.0
