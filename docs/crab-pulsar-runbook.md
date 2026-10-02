# Crab folded-observation runbook

This is the operational path for one coherent Crab beam from four FPGAs and
40 coarse channels (176--215). It produces a DSPSR archive with 10-second
subintegrations. It is intended for a folded detection, not absolute TOAs.

## What is fixed by default

- ibverbs direct-to-packet-ring capture on
  `enp94s0np0,enp134s0np0,enp175s0np0,enp216s0np0`
- capture CPUs `1,11,13,15`; worker CPUs `3,4,5`
- 32-way/27 oversampled fine PFB, two edge bins clipped on each side
- 1120 internal calibrated bins compacted to 1080 DADA channels (original GPU bins 3–29 per coarse channel)
- A8/A12 polarization reversals from `stream_antenna_map.json`
- A14 and A36 excluded; A1-XX excluded by calibration metadata
- one GPU DADA ring: 300 blocks of 8,855,552 bytes (5.308416 seconds)
- DSPSR: 256 phase bins and 10-second subintegrations

The A5<->A29 and A11<->A13 identity swaps are experimental and are never
enabled unless `--identity-swaps` is explicitly supplied.

## Before each run

Confirm all four FPGAs are tuned to 40 channels centred on 196 (176--215):

```bash
cd /data/BLACKMESA_0/lambda/scripts
./configure-fpga.sh -a 0,1,2,3 -n 40 -f 196 -m tune
```

After every rebuild, restore the raw-packet capability:

```bash
sudo setcap cap_net_raw+ep \
  /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps/pulsar_fold_4_40
```

The launcher refuses a real run if this is absent. It also refuses to take
ownership of a pre-existing DADA key `0xdada`; inspect such a ring instead of
blindly deleting it.

Run the launcher outside any restricted filesystem/IPC sandbox. A restricted
diagnostic invocation was observed to kill the persistent `dada_db -w` process
immediately after ring creation, before DSPSR could attach.

The launcher runs `dada_db -w` in a detached process session and verifies both
its PID and IPC key before starting DSPSR. GPU-backed DADA rings require
persistence mode; do not replace this with non-persistent creation.

## Smoke test

From the CUDA repository:

```bash
./scripts/run_crab_pulsar.sh --duration 60 --name crab_smoke_zc_retry
```

A good smoke must show all of the following:

1. `ibverbs receive mode: direct into packet ring (no staging memcpy)`
2. Received and Processed both growing (not Received=6144, Processed=0)
3. Pipeline Runs Queued growing
4. `PSRDADA block size validated: 8855552 bytes`
5. normal pipeline shutdown and DSPSR exit after EOD
6. at least one `crab*.ar` in the printed run directory

The pipeline publishes the DADA header before CUDA/PFB warm-up. DSPSR has
taken between about 14 and 37 seconds to prepare this configuration, so GPU
warm-up duration is not a reliable proxy for readiness. The launcher waits
for DSPSR's explicit `prepared in` message before the application opens its
capture sockets.

Inspect the archive using the local PSRCHIVE tools, for example:

```bash
/data/BLACKMESA_1/sma075/psrdada-local/bin/psrstat \
  -c length,nsubint,nchan,nbin build/pulsar_runs/crab_smoke_zc_retry/crab.ar
```

The exact archive filename is printed by the launcher. Review
`pulsar_fold.log` and `dspsr.log` in the same directory before continuing.

## Fifteen-minute fold

Only after the smoke succeeds:

```bash
./scripts/run_crab_pulsar.sh --duration 900 --name crab_15min_normal
```

The output archive contains approximately 90 ten-second subintegrations,
which can be frequency/subintegration zapped for RFI before summing.

For an identity-map A/B comparison, use a separate short run:

```bash
./scripts/run_crab_pulsar.sh --duration 60 --identity-swaps \
  --name crab_smoke_identity_swaps
```

Do not replace the normal 15-minute run with the experimental map unless its
folded S/N is demonstrably better.

## Calibration and ephemeris caveats

`build/apps/crab_fine_gains_window006.json` is per fine channel and is applied
per fine channel. It is generated from the accepted CASA K/B/phase/G tables by
`scripts/export_casa_fine_beam_gains.py`; the C++ loader conjugate-inverts the
measured Jones gain. Never substitute the old coarse `weights.json`.

`config/crab_jodrell_2026-07-24.par` is newer than the installed 1991 psrcat
spin solution and is adequate for this short detection attempt provided no
unmodelled Crab glitch occurred after July. The packet stream lacks a proven
absolute UTC-to-sample epoch, so the resulting archive is not an absolute-TOA
product. That does not prevent a stable 15-minute fold; it changes the
arbitrary displayed pulse phase.

For a future 24-channel high-band experiment, tune the FPGA backend to 296--319
and pass both an independently accepted gain file and the channel start:

```bash
./scripts/run_crab_pulsar.sh --duration 600 --coarse-channels 24 \
  --min-channel 296 --output-mode intensity \
  --gains /absolute/path/to/approved_296_319_fine_gains.json \
  --name crab_240mhz_freshcal_10min
```

The launcher validates gain frequency provenance, so the old low-band file is
deliberately rejected for this command. The 24-channel unique fine-frequency
axis must contain 649 bins spanning 230.859375--249.609375 MHz; the internal
GPU processing axis contains 672 bins because coarse-boundary bins are
duplicated.

As of 2026-09-23 there is **no approved 296--319 gain file**. The fresh dawn
capture exported correctly, but the all-sky, Tau A, and solar calibration
trials failed solution-coverage/coherence QA. In particular,
`build/calibration_runs/crab_240mhz_20260923/crab_240mhz_fine_gains_clean31.json`
is only a schema test artifact and must not be used for an observation. See
`docs/ingest-optimization-handoff.md` for the exact failure metrics. Do not
weaken or bypass the launcher provenance checks to work around this.

## Known failed run

`build/pulsar_runs/crab_smoke_20260923` contains no valid observation. It
identified the missing ibverbs zero-copy arming call: the ring stopped at
Received=6144, Processed=0 and no archive was written. Keep it only as failure
evidence.

`build/pulsar_runs/crab_machinery_early_header_15s_retry` is also not an
observation. It proves that the early header works (`dspsr: prepared in
17.5843 seconds` before capture), but the diagnostic kernel backend supplied
only about one second of usable beam data, shorter than one requested
10-second subintegration, so no archive was expected.
