# Synthetic pulsar replay validation

This checks the real four-FPGA PCAP ingest, fine PFB, inverse antenna gains,
casacore RA/Dec steering, DADA output, and DSPSR folding in both output modes.
It uses one coarse channel so a 12-second fixture is about 1.86 GB.

## Reproduce

From the repository root, build the replay targets and pointing helper:

```bash
cmake -S . -B build
cmake --build build --target pointing_enu \
  pulsar_fold_replay_intensity_4_1 pulsar_fold_replay_voltage_4_1 -j2
python3 scripts/create_pulsar_replay.py \
  --output-dir build/pulsar_replay_casacore --duration 12
./scripts/run_pulsar_replay.sh build/pulsar_replay_casacore intensity
./scripts/run_pulsar_replay.sh build/pulsar_replay_casacore voltage
```

The generator writes `pulsar.pcap`, `measured_fine_gains.json`,
`zero_fpga_delays.json`, `synthetic.par`, and `truth.json`. It uses the
repository's Crab RA/Dec and antenna positions. `pointing_enu` calls the same
casacore conversion as the pipeline at the fixture's `steering_utc`, so its
injected carrier phases and fractional baseband delays match the steering
weights. The PCAP sample counter starts at 1024 because the parser rejects
zero. That counter offset does not change the pulse phase or duration.

The synthetic pulsar has a 0.5-second period. Absolute pulse phase is free;
DSPSR's `UTC_START` uses the replay's wall time. The tiny positive DM in the
synthetic ephemeris satisfies DSPSR's coherent-voltage setup and has negligible
dispersion over this one-channel band. The launcher validates the gain schema,
creates an isolated run directory, waits for the DADA consumer, and cleans up
only the ring it created.

## Result on 2026-09-23

The 12-second PCAP contained 694,448 packets. Both final runs processed all
694,448 with `Missing=0` and produced 28-channel, 128-bin archives of about
11.997 seconds with two subintegrations:

| Mode | DSPSR backend | Folded S/N | Archive |
| --- | --- | ---: | --- |
| Detected intensity | GPU | 1346.6 | `build/pulsar_replay_casacore/intensity_final/pulsar.ar` |
| Complex voltage, PP+QQ (`-d 1`) | GPU | 1554.3 | `build/pulsar_replay_casacore/voltage_sum_final/pulsar.ar` |
| Complex voltage, separate PP/QQ (`-d 2`), diagnostic | GPU, patched DSPSR | 930.1 | `build/pulsar_replay_casacore/voltage_gpu_final/pulsar.ar` |
| Complex voltage, separate PP/QQ (`-d 2`), diagnostic | CPU | 1141.2 | `build/pulsar_replay_casacore/voltage_cpu_run2/pulsar.ar` |

The requested voltage product is summed polarization (`PP+QQ`, DSPSR `-d 1`).
The earlier `-d 2` runs were an unnecessary choice to retain separate PP and QQ
profiles; they are kept only as a diagnostic of the DSPSR GPU bug. A pre-patch
GPU `-d 1` run on the saved DADA voltages already folded normally (S/N 1558.4), so the
requested `-d 1` path does not need the DSPSR patch. The profiles have broad
peaks; their absolute phase is arbitrary. On the first 1024 packet groups,
applying the inverse measured gains
and the known steering phase gave 10.13 times the coherent beam power of
steering alone.

The original GPU two-polarization voltage fold produced a huge single-bin
artifact, despite finite DADA voltages and a normal CPU fold. CPU and GPU
binary dumps matched exactly before convolution and to floating-point
roundoff just before detection (RMS difference 1.29e-4 against RMS 1131).
The cause was in-place CUDA square-law detection: parallel GPU writes of
float power samples could overwrite unread complex voltage samples in FPT
order. The optional `patches/dspsr-cuda-detection-out-of-place.patch` fixes
that `-d 2` mode by giving GPU detection a distinct output time series. The
patched GPU DSPSR folds the saved DADA voltage file with S/N 1739.2 and the
full PCAP replay with S/N 930.1. The launcher uses GPU DSPSR with `-d 1` for
the requested summed voltage product. The raw voltage data itself has a 12.87
peak-to-median power ratio at the injected period.

The original SourceForge `master` was still at `3f3d5b43` (27 May 2026)
when checked on 23 September 2026; it is the unpatched base of the local
DSPSR checkout. The [DSPSR project site](https://dspsr.sourceforge.net/)
says that development has moved to GitLab. The official DSPSR GitLab `main` at commit
[`421f1fd4`](https://gitlab.com/ska-telescope/pst/ska-pst-dspsr/-/commit/421f1fd4551d87870df4a5e749f00ba5140db361)
(15 September 2026, checked 23 September 2026) still selects an in-place
detector for `-d 2` and dispatches the same CUDA FPT PPQQ kernel. This is a
source comparison; the upstream binary was not run here. The tested local
fix remains needed for this configuration.

The illustrated report is `docs/pulsar-replay-report.tex` and its compiled
PDF is `docs/pulsar-replay-report.pdf`. Regenerate its eight PDF figures with
`python3 scripts/plot_pulsar_replay.py`, then compile from `docs/` with
`pdflatex pulsar-replay-report.tex` (twice for references).
