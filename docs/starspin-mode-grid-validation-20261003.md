# Starspin 1–4 FPGA mode check (2026-10-03)

The four 40-channel modes were run for 60 seconds each through
`scripts/run_starspin.sh` with the `ibverbs` capture backend. The observing
window was channels 104–143. These live runs used the capability-bearing
`/data/BLACKMESA_0/lambda/starspin/build/apps/starspin_N_40` binaries via
`--binary`; the freshly rebuilt binaries in this checkout did not yet have
`cap_net_raw=ep`. The launcher, input files, CPU affinity, and output directory
were from this checkout. These results therefore validate the shared binaries
and launcher, not a live run of the latest local executables.

## FPGA and delay state

All four boards were rebooted together and reported DONE HIGH. Full eight-channel
initialization at 116–123 reported five healthy ADC groups per board. They were
then tuned to 104–143, with that range read back on each board. Eight channels
(116–123) were captured from the final 40-channel stream. All six pairwise
packet-delay measurements closed exactly (maximum residual 0 samples). The
delay file used for the final runs is
`build/starspin_runs/session_20261003_modes40/delays_40ch_all_reboot.json`.

The first attempt required investigation: FPGA 3 initially had ADC sync errors
and low, unscaled samples. Reapplying its ADC settings without reinitializing
restored its ADC sync and signal levels. Rebooting only FPGA 3 later left it
about 508 million samples away from the other boards; the four-FPGA correlator
then discarded nearly all packets and queued no GPU pipeline work. The
coordinated reboot and new delay solve resolved this. The failed attempt and
all diagnostic logs are retained in the session directory.

The commissioning checkout still had the legacy `-m init-tune`/`tune` interface
and invoked the upstream `uconsole`, which now presents a platform menu. A
session-local copy of that script selected `lambda_uconsole` instead. No
commissioning source file was changed here. When the new comma-separated
`-m init,adc,tune` interface lands, this workaround should be retired.

## Final 60-second runs

| FPGAs | Received | Processed | Missing | HDF5 blocks with missing packets | Pipeline runs |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 | 34,717,884 | 34,717,884 | 6,156 | 44 | 3,391 |
| 2 | 69,446,384 | 69,446,384 | 1,776 | 0, 2 | 3,391 |
| 3 | 104,166,089 | 104,166,062 | 6,615 | 0 | 3,391 |
| 4 | 138,894,198 | 138,894,191 | 3,870 | 0, 4, 6 | 3,391 |

All runs exited cleanly. `Discarded`, `FutureQueued`, `StuckUnprocessed`, and
`NICDrops` remained zero. Sampled HDF5 visibilities were finite and nonzero,
and integration sequence numbers increased. The HDF5 `vis_missing_nums`
dataset confirms the affected blocks above. Packet loss was **not strictly
zero**: the one-FPGA run had a 6,156-packet burst around block 44, and the
other runs lost small numbers of packets near startup. The 2- and 3-FPGA
runs had no additional missing packets after their first reported interval;
the four-FPGA count reached its final value in the first two intervals.

The 40-channel binaries use one shared, interleaved receive ring. Capture CPUs
are local to their NICs (CPU 1 on NUMA node 0; CPUs 11, 13, and 15 on node 1),
but the ring and packet workers are on node 0. The launcher's per-NIC ring NUMA
settings are used only by binaries compiled with `SPATIAL_PER_NIC_RING`; these
40-channel targets were not. These runs do not test NIC-local per-NIC rings.

The full logs, PCAP calibration captures, readbacks, HDF5 files, and machine-
readable summary are under `build/starspin_runs/session_20261003_modes40/`.
