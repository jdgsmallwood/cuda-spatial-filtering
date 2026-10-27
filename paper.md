---
title: 'Starspin: A GPU correlator for LAMBDA telescope commissioning'
tags:
  - C++
  - CUDA
  - radio astronomy
  - interferometry
  - telescope commissioning
authors:
  - name: Justin Smallwood
    orcid: 0009-0006-4111-7610
    affiliation: "1, 2"
  - name: Adam Deller
    affiliation: "1, 3"
  - name: Andrew Jameson
    affiliation: 3
  - name: Chris Phillips
    affiliation: 2
affiliations:
  - name: Centre for Astrophysics and Supercomputing, Swinburne University, VIC 3122, Australia
    index: 1
  - name: CSIRO, Space and Astronomy, P.O. Box 76, Epping, NSW 1710, Australia
    index: 2
  - name: Fourier Space, Swinburne University, Hawthorn, VIC 3122, Australia
    index: 3
date: 15 October 2026
bibliography: paper.bib
---

# Summary

Radio interferometers combine signals from multiple antennas to measure the sky.
A correlator computes pairwise products of those signals, producing complex
measurements called visibilities. During telescope commissioning, these
measurements also reveal problems in signal routing, timing, and calibration.
Starspin is an open-source GPU correlator used for commissioning the LAMBDA
telescope (Low-frequency Australian Megametre-Baseline Demonstrator Array).
It converts network packets from FPGA receivers into integrated,
full-polarization visibilities, with optional fine frequency channelization.
It records the configuration and packet-quality information needed to interpret
the output. Recorded packet files can be replayed through the same processing
path, allowing numerical checks without access to an operating telescope.

Starspin is implemented in C++ and CUDA. It combines instrument-specific packet
handling and delay alignment with existing GPU signal-processing libraries.
Its primary output is HDF5, suitable for subsequent inspection and calibration.
The repository also includes companion beamforming and spatial-filtering
applications; this paper concerns the Starspin correlator and the shared
infrastructure required to operate and validate it.

# Statement of need

Commissioning requires more than calculating correlations quickly. Engineers
must establish that receiver streams reach the correct antennas and
polarizations, that packet counters and delays align their samples, and that
frequency coordinates match the channelizer. A plausible-looking visibility
can still be wrong if any of these conventions are inconsistent. Conversely,
missing packets or a delay solution from an earlier FPGA boot can make a
correct implementation appear faulty.

Starspin addresses this need for the LAMBDA commissioning team by bringing
packet decoding, stream mapping, delay correction, correlation, and output
provenance into one executable workflow. It supports both the legacy LAMBDA
packet format and LAMBDA CODIF packets. Its intended users are telescope
commissioning researchers and developers building related GPU back ends.
Instrument-specific formats and calibration conventions are explicit parts of
the interface, rather than assumptions hidden in downstream analysis.

The software accepts externally determined gains and delay solutions; it does
not make an arbitrary capture scientifically calibrated by itself. Live use
requires a verified FPGA boot state, matching packet delays, and board channel
readback. Separating these operational prerequisites from numerical checks
makes failures easier to diagnose and avoids attributing instrument-state
errors to compute performance.

# State of the field

GPU correlation is established practice. xGPU implements an efficient GPU
X-engine for radio-astronomy correlation [@clark2013]. ASTRON's Tensor-Core
Correlator (TCC) supplies the correlation library used by Starspin [@tcc].
Bifrost provides a broader Python/C++ framework for constructing streaming
astronomical pipelines [@cranmer2017]. These projects address different layers:
a correlation kernel, a reusable accelerator library, and a general streaming
framework, respectively.

Starspin's contribution is the executable integration of LAMBDA packet and
calibration semantics with those numerical operations and an auditable output
format. It reuses TCC and ASTRON's gpu-filter polyphase filterbank [@gpufilter].
A separate instrument application allows receiver routing, boot-dependent
alignment, and capture provenance to evolve without embedding LAMBDA-specific
policy inside a general correlator library. A general framework could also
host these operations, but would still require the instrument adapters and
verification work. Starspin therefore provides a concrete commissioning
workflow rather than claiming to replace existing frameworks or to introduce
a new correlation algorithm. Performance comparisons must distinguish these
layers and use matched workloads.

# Software design

The processing path comprises packet capture, host reassembly, GPU preparation,
optional fine channelization, correlation, integration, and output. Compile-time
configuration fixes the principal data dimensions and selects executable
variants for different FPGA and coarse-channel counts. This makes array shapes
consistent across stages and enables specialized GPU operations, at the cost
of rebuilding when those dimensions change. Runtime inputs select packet
format, frequency range, calibration files, stream maps, and integration length.

The capture interface supports offline PCAP replay and live capture. Reassembly
aligns packet samples and tracks missing input before handing buffers to the
GPU. The Starspin pipeline uses completion events to release host buffers after
transfers have finished. This ownership boundary permits overlap between
capture and computation while preventing reuse of memory still being read by
the GPU. The correlator-only path avoids scheduling beamforming or
eigendecomposition stages that commissioning visibility capture does not need.

Fine-channelized operation uses a 32-bin filterbank and retains 27 distinct
frequency centres per coarse channel. The output records the retained-bin
geometry, receiver-pair ordering, and polarization mapping. The filterbank
currently zero-pads its look-ahead at each processing-block boundary; the
independent reference includes that boundary convention. Coarse-only targets
provide a separate path without this additional channelizer. These choices
make channelization and boundary behavior testable rather than implicit.

Verification combines synthetic packet tests, numerical invariants, and a small
synthetic-packet replay. The replay reference independently decodes CODIF and
computes the filterbank and visibility products with NumPy, including voltage
rounding and integration geometry. It checks all output visibility components
against fixed tolerances and verifies that conjugation and missing-scale
negative controls are distinguishable. Such checks establish numerical
agreement for the exercised configuration; they do not establish live network
throughput or calibration validity for a different observing session.

A separate coarse-only check compares Starspin with the `zerocorr` utility
distributed with DiFX 2.9.0 [@deller2007]. Seeded synthetic complex voltages pass
through CODIF replay in Starspin and lossless VDIF repacking for DiFX. All
baselines and polarization products are compared over identical sample-counter
intervals, with an analytically derived normalization and no fitted gains or
phases. This checks coarse-channel correlation; it does not validate DiFX's
distributed `mpifxcorr` application, inter-board delay calibration, or the fine
filterbank. The latter retains its separate NumPy reference.

# Research impact statement

Starspin is in use for commissioning the LAMBDA telescope. Its research role is
to provide visibility data and diagnostics while the telescope's signal chain
is being established. Site-specific observing procedures, calibration solutions, and commissioning
records are maintained outside the public software repository. Public validation
uses a reproducibly generated synthetic fixture, avoiding redistribution of
site configuration or commissioning samples.

The commissioning workflow links software validation to instrument state:
operators retain the delay solution belonging to the verified boot and record
it alongside capture configuration. For reuse outside that site, the bundled
CODIF fixture and numerical reference provide a smaller, telescope-independent
entry point. The public source includes tests, contribution guidance, and
support through the issue tracker. The accompanying reviewer guide distinguishes
portable checks from GPU and instrument-dependent verification. These materials
support assessment and adaptation of the processing path without claiming
adoption by other observatories or results from an astronomical observing
campaign.

# AI usage disclosure

Claude Code (Claude Sonnet 4.5, Opus 4.5, and Fable 5) and OpenAI Codex
(GPT 5.4, 5.5, and 5.6-Sol) assisted in creating the LAMBDA signal-processing
pipeline. The lead author reviewed that code and validated it with the test
suite. OpenAI Codex also assisted with the present submission's build fixes,
verification workflow, documentation, and paper draft. Preparation checks
include automated Python tests, GPU tests, and an independent visibility
reference. AUTHOR CONFIRMATION REQUIRED: confirm the model identifier used for
this preparation and human review of the final submission changes, including
responsibility for the scientific and design decisions. Automated checks do
not substitute for that final author review.

# Acknowledgements

The lead author is funded by the Commonwealth Scientific and Industrial
Research Organisation (CSIRO) Industry PhD program, in partnership with Fourier
Space. We acknowledge the Gomeroi people as the Traditional Owners of the land
on which the Narrabri station of LAMBDA is situated. We also acknowledge the
engineers and scientists who have contributed to LAMBDA, and the maintainers of
TCC, gpu-filter, and the other libraries on which Starspin depends.

# References
