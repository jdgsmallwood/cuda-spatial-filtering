# Starspin colleague startup — 2026-10-01

All four FPGAs were rebooted (DONE pins HIGH), fully initialized (20/20 ADC groups OK), and calibrated on eight coarse channels. The first 192–199 capture was too short for the expanded delay window; a longer capture still had insufficient common signal for FPGA 0–2. Retuning to 116–123 produced all six pairwise delays with **zero sample closure residual**. No further reboot was performed.

The initial all-board scale check found 240,000 scale values per board, all 1, and 95–96% nonzero active IQ components; no 64/128 anomaly. Final register readback confirmed **288–327 on all four boards**.

Current delay solution: `build/starspin_runs/alveo_delays_20261001_colleague.json`. Both launchers default to this file. It is valid for this boot state; a later reboot requires a fresh calibration. Changing the channel range alone does not replace the delay file.

## Capability helper

```bash
cd /data/BLACKMESA_0/lambda/starspin
scripts/grant_capture_caps.sh --dry-run
scripts/grant_capture_caps.sh
```

The helper invokes sudo and applies exactly `cap_net_raw=ep`. It discovers all built `starspin_1_*` through `starspin_4_*` ELF executables, including diagnostic/experimental variants and resolved aliases. It currently finds 32 binaries in the shared checkout (24 in development). Re-run it after rebuilding/replacing binaries. It replaces the stale wrapper listing only old starweave names; it does not build missing channel variants or change other applications' capabilities.

## Run the correlator

```bash
cd /data/BLACKMESA_0/lambda/starspin
source scripts/env.sh
scripts/check_env.sh
scripts/run_starspin.sh --channels 40 --min-channel 288 \
  --configure-fpga \
  --delay-file build/starspin_runs/alveo_delays_20261001_colleague.json \
  --duration 60 --name "test_$(date +%Y%m%d_%H%M%S)"
```

Add `--dry-run` to validate and print the command without tuning or capturing. The launcher uses **ibverbs into CPU packet rings**, then the normal full GPU pipeline. `--configure-fpga` checks/tunes the selected boards to match the requested channel range; without it, `--channels` and `--min-channel` only configure the software. Results are in `build/starspin_runs/`.

## Verified run

The exact shared-checkout path and sourced environment above passed a 30-second full-pipeline run named `colleague40_verified_20261001`:

- Approximately 69.45 million packets received and processed.
- `Discarded=0`, `FutureQueued=0`, `StuckUnprocessed=0` throughout the reported samples.
- Missing remained at 5,168 after startup; no missing packets in HDF5 blocks 3–31.
- 31 HDF5 blocks, 36 antennas, 666 baselines, continuous sequence ranges (end denotes the final 64-sample packet's start).
- 19,840 sampled visibility components were finite, with nonzero output; the whole visibility array was not scanned.
- Clean shutdown. No capture remains running; the boards are left streaming 40 channels, 288–327.

Final counters are sampled concurrently, so small received/processed differences are not an exact accounting discrepancy:

```
Stats: Received=69447342, Processed=69447347, Missing=5168, Discarded=0, FutureQueued=0, StuckUnprocessed=0, NICDrops=0
```

The shared and development 40-channel executable hashes matched before the test; both already had `cap_net_raw=ep`. Shared `check_env.sh` passed. The colleague's original stall was not reproduced, and its cause is not established. Mismatched FPGA/software tuning and delays from an earlier boot are concrete setup conditions to check first; the command above explicitly handles the tuning and selects the fresh delay solution.

## Evidence

Session logs, channel readback, validation JSON, and retained calibration PCAPs:
`/data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/starspin_runs/session_20261001_colleague/`.

The shared checkout contains copies of the session logs/JSON, the delay file, this guide, and the tested HDF5/log outputs. Calibration PCAPs remain in the development session directory.
