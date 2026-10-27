# External deployment configuration

Keep hardware addresses, physical antenna maps, coordinates, calibration,
credentials, and observing records outside this repository. `config.example.json`
is a synthetic loopback example, not a telescope configuration.

Pass the site's JSON using `--config-file`. It may contain `array_location`,
`frequency_plan`, `antenna_positions`, and `network_interfaces`. Optional
`antenna_mapping` maps FPGA-ID strings to arrays of ten physical receiver IDs;
negative IDs denote disconnected receivers. Every selected FPGA must be present
when that mapping is supplied. Without a mapping, output uses logical receiver
indices, which must not be interpreted as physical antenna IDs.

Use `--stream-antenna-map` for explicit per-polarization wiring and canonical
ordering, `--gains` for gains, and `--delay-file` for a calibration verified for
the current FPGA boot. Preserve boot state, channel readback and delay closure
with each live observing session. Do not substitute synthetic zero delays for
site calibration. Production receiver pruning requires its explicit stream map.

The optional `scripts/run_starspin.sh` launcher loads a trusted shell profile
named by `STARSPIN_SITE_PROFILE`. Set `CONFIG_FILE`, `STREAM_MAP`, `INTERFACES`,
`CAPTURE_CPUS`, `WORKER_CPUS`, `PIPELINE_CPUS`, and `CUDA_ROOT` there; supply the
current `--delay-file` explicitly. `CONFIGURE_SCRIPT` is an external site-owned
hardware command and runs only with `--configure-fpga`. Review a profile before
sourcing it: it is shell code, not an inert data file.

PSRDADA header generation requires `--telescope`; the replay shell script reads
`TELESCOPE`, `PSRDADA_BIN`, `TEMPO2_ROOT`, and `CUDA_HOME` from the environment.
The public software does not supply an observatory identity or installation path.

Commissioning runbooks and one-off observing analyses have moved to the local
private deployment archive. Generic analysis routines, wire-format implementations,
synthetic tests, and paper attribution remain public. Existing site scripts may
need paths adjusted after migration. Saved output embeds configuration for
provenance, so inspect it before sharing it publicly.

Removing tracked files changes the working tree, not existing Git history,
previous releases, clones, or ignored local artifacts. No history rewrite has
been performed. Coordinate any required historical cleanup separately.
