
## Running marimo

```
bash
source .venv/bin/activate
marimo edit
```

Take note of the port it's running on - somewhere around 2718

ssh to example-host and forward that port to your computer
ssh -L <port>:localhost<port>  example-host

for example:
```
ssh -L 2718:localhost:2718 example-host
```

## Running four-FPGA Starspin

Load a trusted external deployment profile and supply delays verified for the
current boot:

```bash
STARSPIN_SITE_PROFILE=/path/to/site.sh scripts/run_starspin.sh \
  --delay-file /path/to/current-delays.json --duration 600
```

See [deployment configuration](../docs/deployment.md). Use `--dry-run` to inspect
the command. The public repository does not contain production NIC assignments,
CPU affinities, antenna wiring, calibration, or operational runbooks.

For local socket-path testing, follow [kernel loopback](../docs/kernel-loopback.md).
For reproducible coarse-only comparison against DiFX's `zerocorr`, use
`check_difx_visibility_parity.py` following [DiFX verification](../docs/difx-validation.md).
For a validated coarse-PFB amplitude table, pass `--deripple-config` and the
external JSON path after the launcher's `--` separator. This corrects voltages
before correlation and does not modify firmware.
