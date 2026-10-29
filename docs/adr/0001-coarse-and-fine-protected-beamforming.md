# Compare coarse and fine decomposition with common fine beam outputs

Both adaptive beamforming paths produce the same fine-channel beam products.
The coarse path correlates the samples entering the fine channelizer and
derives a coarse spatial filter, while the channelizer supplies fine beam
voltages; it performs no fine-channel correlation. The fine path correlates
and decomposes fine-channel data. Combining fine covariances would retain the
fine correlation cost and would not test the intended alternative.

The initial comparison uses fixed nulling ranks 0, 1, 2, and 4, with matched
covariance integration durations and filter-update cadence. Both paths are
compared with and without the rank-1 source protection described in Smallwood
et al., submitted, Paper1, “Protecting Unresolved Point Sources”: restrict the
covariance to the protected signature's orthogonal complement before selecting
the nulled eigenmodes. Calibration, whitening and the voltage-domain filter
must use consistent coordinates.

For the initial coarse trial, protect the Sun's signature at the coarse-channel
centre and reuse the filter across that channel's fine bins. Measure solar
retention in every fine bin; centre-frequency protection is not a guarantee
across the passband. Fine protection uses each fine channel's solar signature.
Use prior calibration and observation-time solar steering; online calibration
is a separate investigation.

Diagnostic captures save pre-whitening visibilities at the decomposition
cadence for offline eigenanalysis and rank-selection research. Live eigenmode
telemetry is deferred. Benchmark sustained successfully processed input payload
GB/s with the intended beam output, reporting packet loss and diagnostic
visibility-recording overhead separately. Compare before/after RFI spectra
and solar retention locally and in production.

## Application and output

Include a weights-only beamforming baseline that bypasses correlation,
eigendecomposition and spatial-filter construction/application entirely, while
retaining the same fine channelization, steering/calibration, beam count and
complex-voltage output workload. Compare its sustained input throughput with
both adaptive paths to quantify their total added cost. Rank-zero adaptive
operation is a separate reference, not a substitute for this baseline, because
it may still perform correlation and eigendecomposition. Keep paired-output
writing volume matched for the benchmark, using identical unfiltered outputs
for the weights-only case.

Add a dedicated adaptive beamforming app with its own GPUPipeline implementation,
reusing production ingest, writers and existing GPU stages. Coarse and fine
decomposition are configurations of the same pipeline, sharing beamforming,
protection and output code. Preserve existing compile-time dimensions where
needed; use runtime selection where practical. Keep the existing correlation-only
Starspin interface intact. Verify the production hardware and configuration
before selecting benchmark dimensions; keep deployment details local.

Begin with one Sun-tracking beam, processing the two polarizations separately,
and write paired unfiltered/filtered complex voltages by default. Derive power
spectra offline. Include the voltage writer in end-to-end throughput measurements.
Use one common covariance duration, initially the existing adaptive block duration
after verifying it for the selected packet format, and update once per block.

## Live comparison

The `adaptive_beam` executable selects `--decomposition weights-only|coarse|fine`,
`--null-rank N`, `--sun-protection off|rank-one`, and an explicit
`--calibration unity|gains`. Unity mode needs neither gains nor an eigenvalue-count
file. Starspin also skips both files because its correlation-only path does not
use them. Existing filtering applications retain their file-loading behavior.

Use `--record-visibilities --vis_output_file PATH` to save the raw packed
covariance once per decomposition block, before whitening. This is unavailable
in weights-only mode. The saved matrix is a sum, not a sample mean; its metadata
records samples per covariance and voltage scale. Diagnostic writing adds
transfer and storage work, so benchmark it separately from beam-only writing.

Fine-channel beam voltages use the production channelizer's pre-fp16 scale of
1/64 to avoid overflow. The beam file records `fine_voltage_scale`; divide stored
voltages by that scale to recover the unscaled channelizer voltage convention.
Fine covariance sums carry its square, while coarse covariance sums have unity
voltage scale. This conversion alone does not establish physical flux units.

Use repeated, short, interleaved coarse/fine and protected/unprotected trials.
Each trial has simultaneous unfiltered/filtered beam outputs. A small packet
capture is optional: inspect sequence continuity and packet loss before using it.
Between-trial RFI variability remains a limitation of live comparisons.
Reuse verified FPGA boot state and matching delays throughout an observing
session; follow AGENTS.md for verification, tuning and capture provenance.

## Calibration fallback

Solar steering uses observation UTC, site location, antenna positions and channel
frequencies. Validate the unfiltered solar beam before assessing mitigation.
Replay uses capture time rather than current wall-clock time.

When gains are unavailable, offer geometric Sun protection with unity gains,
unprotected operation, or abort. Unattended runs require an explicit choice.
Record the chosen mode: geometric protection preserves an assumed signature and
does not establish preservation of the actual solar signal.

For partially valid calibration, exclude affected receivers/channels where
feasible and report the exclusions; require an explicit decision before falling
back to unity gains. Do not silently mix calibrated and assumed responses.
Missing geometry, mapping or usable timestamps prevents meaningful geometric
Sun protection, but need not prevent an explicitly unprotected throughput test.

## Follow-on research

### Fixed RA/Dec tracking validation

Include fixed J2000 RA/Dec beams in the pointing follow-up. Their celestial
coordinates stay fixed, but their local ENU direction and beam weights must
advance with observation UTC. Use geodetic up consistently with the antenna
positions for both RA/Dec and solar tracking. Check independent reference
directions at multiple epochs and the resulting weights uploaded to the GPU.

Before claiming observational validation, verify tracking on a calibrated
astronomical source over time, including both polarizations and the processing
frequency grid. Audit capture-time advancement in the older replay applications;
the new adaptive pipeline's replay clock alone does not validate those callers.
Record the Earth-orientation table versions with that validation.

Use diagnostic visibilities to evaluate automatic nulling-rank rules offline
before selecting a live policy. Investigate online calibration using astronomical
and interfering sources separately from the initial implementation. Numerical
solar-loss acceptance thresholds remain to be established from calibration and
measurement uncertainty; model preservation alone is not observational validation.

## Local implementation and validation

The new app defaults to one-second steering updates (override with
`--steering-update-interval-seconds`). Replay interprets `--steering-utc` as the
UTC of the first processed input block's first sample, and evaluates steering at
block midpoints using sample sequence offsets. The shared clock is used by the
older observing, beamformed-bandpass, adaptive-bandpass and pulsar-fold paths as
well; provide the capture UTC when replaying tracked beams.

`--calibration ask` is the default. If the gains file exists it is used; otherwise
an interactive terminal offers geometric protection, unprotected unity gains or
abort. Unattended missing-gain runs must select `gains` or `unity` explicitly.
Malformed calibration schemas abort. Missing, invalid and explicitly excluded
receptors receive zero beam weight and are masked from filter estimation and
application. Coarse mode propagates each exclusion across the whole coarse
channel. The app rejects ranks that leave no active input, records exclusions
in both output files, and distinguishes assumed geometric protection from a
prior-calibrated signature. Validation occurs before output files are opened.

`scripts/benchmark/adaptive_beam.py` runs a deterministic synthetic PCAP through
the real app, with paired complex-voltage writing enabled. It compares the
weights-only baseline and coarse/fine ranks 0, 1, 2 and 4, both protected and
unprotected, in a shuffled order over repeated passes. Additional rank-two
protected trials record covariances to measure diagnostic-writing overhead.
The numerator is successfully processed 8-bit complex input payload, excluding
packet headers, scales and duplicated boundary samples. The denominator spans
capture start through GPU completion and writer flush; it excludes allocation
and startup warmup. The script independently checks the numerator against the
written arrival flags and checks every processed block was written. Flushing
includes the OS page cache, not an fsync guarantee. Replay loss and discarded
packets are reported; these results are not live network capacity measurements.

The `bench_adaptive_beam_4_40` target exercises 40 receivers, 40 coarse channels
and 1,120 retained fine channels, using one GPU buffer to fit the local 8 GiB
device. Fine-channel operation does not allocate the unused legacy post-beam
FFT buffers. Weights-only operation does not allocate eigensolver workspace;
protected operation releases the full-space workspace before allocating its
reduced-space workspace. Allocation and solver API failures stop the app.
Per-matrix eigensolver failures produce an identity filter, increment the
recorded `eigensolver_failures` counter and make the run exit unsuccessfully;
such runs must not enter the benchmark comparison.

Weights-only and fine-decomposition operation reuse Starspin's fused raw-packet to
channelizer-input preparation, including scales, delays and stream permutations.
It omits the four intermediate half-voltage buffers used by the staged path.
Beam multiplication remains ccglib's optimized complex GEMM with half inputs and
float accumulation. `bench_weights_pipeline_40 [blocks] [gpu_buffers]` isolates
this GPU path (including host input copies, excluding packet parsing and output
copies/writing); `scripts/benchmark/adaptive_beam.py --weights-only` retains the
full replay and paired-voltage writing workload for end-to-end comparisons.
Use `--gpu-buffers N` on the app or benchmark script to measure overlap at a
chosen memory budget; the app records the selected count. The build default is
unchanged, and additional buffers may exceed device memory in decomposition
modes even when weights-only fits.
