
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

## Running four-FPGA Starweave

Use the self-contained launcher from the repository root:

```bash
scripts/run_starweave.sh --duration 600
```

It defaults to the proven 40-channel range (176--215), ibverbs direct-to-ring capture, the four
production NICs, known-good CPU affinities, and the required configuration/mapping files. Run
`scripts/run_starweave.sh --dry-run` to preview everything without starting a capture, or
`scripts/run_starweave.sh --help` for overrides. Operational details are in
`docs/starweave-capture-runbook.md`; the architecture and 24-to-40-channel changes are explained
in `docs/starweave-24-to-40-explained.md`.

