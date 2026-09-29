
## Running marimo

```
bash
source .venv/bin/activate
marimo edit
```

Take note of the port it's running on - somewhere around 2718

ssh to blackmesa and forward that port to your computer
ssh -L <port>:localhost<port>  blackmesa

for example:
```
ssh -L 2718:localhost:2718 blackmesa
```

## Running four-FPGA Starspin

Use the self-contained launcher from the repository root:

```bash
scripts/run_starspin.sh --duration 600
```

It defaults to the proven 40-channel range (176--215), ibverbs direct-to-ring capture, the four
production NICs, known-good CPU affinities, and the required configuration/mapping files. Run
`scripts/run_starspin.sh --dry-run` to preview everything without starting a capture, or
`scripts/run_starspin.sh --help` for overrides. Operational details are in
`docs/starspin-capture-runbook.md`; the architecture and 24-to-40-channel changes are explained
in `docs/starspin-24-to-40-explained.md`.

For a **validated** runtime coarse-PFB amplitude table, pass it to a
fine-channel binary as `--deripple-config /absolute/path/table.json`. With the
launcher:

```bash
scripts/run_starspin.sh --duration 600 -- --deripple-config /absolute/path/table.json
```

This corrects fine-channel voltages before
correlation, so visibility and power amplitudes scale by the square of the
table. It does not modify firmware. The config is opt-in and must live outside
the repository. The current ROM-only preview is **not** a validated science
table; see `docs/ingest-optimization-handoff.md` for calibration compatibility.
