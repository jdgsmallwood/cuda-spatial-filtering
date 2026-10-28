# Optional Starspin monitoring

Starspin can publish sampled visibility products, capture counters, and Linux
server/process utilisation to Redis, with a provisioned Grafana dashboard.
Monitoring is **disabled by default**. Redis is never required to start, run,
or stop the correlator. Scientific output remains in HDF5.

## Start

From the repository root:

```sh
docker compose -f monitoring/compose.yaml up -d
```

Open <http://127.0.0.1:3001/d/starspin-monitor>. Add these options to an otherwise
working Starspin command:

```sh
--monitor --monitor-port 6380
```

The dedicated stack uses host ports **6380** and **3001**, so it can coexist with
an existing Redis/Grafana installation. Override with `STARSPIN_REDIS_PORT` and
`STARSPIN_GRAFANA_PORT` when starting Compose, and match `--monitor-port`.
When running Starspin in Docker on Linux, use host networking to reach the
published localhost Redis port. The dashboard has read-only anonymous access;
both ports bind to localhost. For remote viewing, use an SSH tunnel, for example
`ssh -L 3001:127.0.0.1:3001 observing-server`, rather than exposing these services.

Stop this optional stack with:

```sh
docker compose -f monitoring/compose.yaml down
```

Redis history is disposable: persistence is disabled. This command does not
stop Starspin or affect HDF5 output. Grafana provisions its dashboard and data
source again on startup. Initial setup needs internet access to pull the pinned
images and install Redis datasource 2.2.0.

## Options and interpretation

| Option | Default | Meaning |
| --- | --- | --- |
| `--monitor` | off | Enable monitoring |
| `--monitor-host` | `127.0.0.1` | Numeric IPv4/IPv6 address; DNS is excluded from the shutdown path |
| `--monitor-port` | `6379` | Use `6380` for the supplied Compose stack |
| `--monitor-interval` | `5` | Seconds between samples; finite, at least one second |
| `--monitor-baselines` | `0:1` | Up to 16 comma-separated receiver pairs `i:j`, with `i <= j` |
| `--monitor-waterfall` | `rx0_p0` | Input whose recent spectral history is retained |
| `--monitor-provenance` | unset | JSON observing record; unspecified verification is `unverified` |

Unusable monitoring settings disable monitoring with a warning and leave the
correlator running; ordinary command-line syntax errors still fail parsing. A valid, positive `frequency_plan.channel_bandwidth_hz` in
Starspin's configuration is required to label spectra. Physical antenna IDs
and receiver mapping are shown in the observation table. Baseline selections
use **output receiver indices**, not physical antenna numbers. Both parallel
polarisation products are published for each selected pair.

Power spectra and band-mean power are raw accumulated correlations, not Jy,
not normalized by integration duration, and not corrected for missing packets.
Spectra average adjacent complex channels into at most 256 bins; phase is taken
after averaging, not averaged as angles. Finite bins from partial integrations
remain visible alongside their integration-level missing percentage. There is
no claim of per-input packet validity. Entirely missing integrations, unused
canonical inputs, non-finite values and zero-amplitude phases are marked invalid.

The Redis plugin mishandles JSON null measurements. Spectral measurements use
numeric strings with `NaN` for invalid values; dashboard transforms convert them
to numbers without replacing them with zero. The waterfall contains at most
120 sampled integrations of the input selected at launch. Change the dashboard's
Input selector to inspect any input's current spectrum and power history;
change `--monitor-waterfall` on the next run to record a different waterfall.

Scalar histories have 24-hour retention. Every run's keys also expire after
24 hours without updates. The separate Redis instance has a 256 MiB storage
limit and rejects additional writes when full; correlation still continues.
The monitor's pending mailbox is one batch, at most 16 MiB, and its waterfall
history is independently bounded. Each run has a unique timestamp/PID namespace;
`starspin:latest` selects the newest published run. Refresh/reopen the dashboard
after starting a new run. The log prints its run ID; an older retained run can
also be selected with the dashboard URL's `var-run` parameter.

The time axis uses **sampling/publication wall-clock time**. Integration start
and end sample counters are recorded separately. These are not inferred UTC
acquisition timestamps, especially during replay. Check the last-monitor-update
panel before interpreting any other panel: ages in cached samples stop updating
when publishing stops. At the default cadence, an update older than 15 seconds
is stale. Output age measures the last successful HDF5 dataset write, not an
`fsync` or durable-disk acknowledgement; batching can make it longer than the
integration age. `pipeline_runs_queued` is cumulative, while `pipeline_backlog`
is the current CPU feeder-queue estimate, excluding already-submitted GPU work.
Capture counters refresh on the existing five-second loop even when spectra
use a shorter interval.

Per-core CPU percentages come from `/proc/stat` interval deltas, excluding guest
time from double-counting. Busy excludes idle and I/O wait; user, system, idle
and I/O wait are also published. The process CPU measurement covers all
correlator threads: 100% means one logical CPU, and values may exceed 100%.
RAM used is `MemTotal - MemAvailable`; process RSS, swap used and swap traffic
are separate metrics. `/proc` may expose host totals from inside a container.
The observation table includes visible cgroup v2 `cpu.max`/`memory.max` values;
missing entries mean unavailable, not unlimited. Cgroup v1 limit discovery is
not implemented.

## Observing provenance

The dashboard does not operate FPGAs or certify calibration. Supply an observing
record containing, for example, `boot_verification`, `boot_id`, `delay_file`,
`delay_closure`, `closure_residual`, and `board_channel_readbacks`. Its contents
are displayed as supplied and recorded with monitoring configuration under
`audit/monitor_provenance_json` in HDF5. The effective delay file path and delay
samples are also recorded. Without a record, verification is explicitly unknown.

Follow `AGENTS.md`: verify boot and matching delays at session start. If either
cannot be verified, reboot the selected boards, run full eight-channel
`init-tune`, solve fresh packet delays and confirm closure before widening the
channel range. Read every board's range before each capture. Reuse verified boot
and delay state for consecutive comparison captures. Do not diagnose a CPU/GPU
or ring bottleneck from unverified observations.

## Failure isolation and legacy writers

The Redis worker has 100 ms connect/socket timeouts, interruptible one-second
retry backoff, and warnings limited to one per 30 seconds. Producers never wait
for Redis and replace stale pending updates. Shutdown drops pending monitoring
work instead of flushing it and interrupts the active socket, including a reply
that keeps trickling data within the socket timeout. A Redis transaction publishes each snapshot and its
sample identity together, so an interrupted transfer cannot expose half a new
snapshot. The visibility output hook copies only selected
products; JSON encoding, `/proc` sampling and Redis transport are off the
scientific writer thread.

`RedisEigendataWriter` and `RedisBeamFFTWriter` now use the same worker. Their
constructors register a setup callback; commands are generated and executed in
the worker, without retaining a large setup queue.
Connection/write failures cannot escape the writer thread. Reconnect recreates
missing series/rules, existing setup is accepted, and duplicate timestamps use
`LAST` for newly created raw series. New eigen and FFT aggregate series retain
24 hours instead of unlimited history. Existing series are not reconfigured.
Their original localhost:6379 default and existing call signatures remain valid;
optional final constructor parameters allow a different numeric host/port.
Their telemetry is now explicitly best effort: overloaded or unavailable Redis
can lose updates, while output buffers keep draining. No scientific HDF5 data
are discarded by this policy.

## Tests and reviewer replay

CPU tests need `g++`, `libhiredis-dev` and `nlohmann-json3-dev`:

```sh
python3 -m pytest -q tests/test_monitor.py
cmake --build build --parallel 2 --target WriterTests
./build/tests/WriterTests
```

These exercise CPU tick arithmetic, memory parsing, spectral amplitude/phase,
invalid data, binning, unavailable/stalled servers, bounded publication and
shutdown, and reconnection. WriterTests covers the actual legacy constructors
and process methods with a port that is reserved but not listening, and the
HDF5 observation/write callbacks. CPU monitor tests run in the Python CI job;
without their compiler/headers, pytest reports explicit skips.

For a known-spectrum end-to-end stack check (no telescope or GPU required):

```sh
g++ -std=c++17 -O1 -pthread -I include tests/test_monitor.cpp -lhiredis -o /tmp/test-monitor
/tmp/test-monitor publish 6380 4
python3 monitoring/check_stack.py
```

This asserts the known power/amplitude values, validity marker, retention,
process memory, waterfall bounds and every dashboard backend query. Run it
against the dedicated test stack; it expects the synthetic monitor fixture.

The numerical correlator replay reuses the existing public CODIF fixture:

```sh
python3 scripts/review_starspin.py --binary build/apps/starspin_1_8 \
  --output-dir build/monitor-off
python3 scripts/review_starspin.py --binary build/apps/starspin_1_8 \
  --output-dir build/monitor-on --monitor
python3 scripts/review_starspin.py --binary build/apps/starspin_1_8 \
  --output-dir build/monitor-absent --monitor --monitor-port 6399
```

Use new output directories and ensure port 6399 is unused. Each run checks every
visibility against the independent numerical reference. Integration counts
vary with timing, so compare matching sequence ranges rather than whole-file
sizes. After the monitored run finishes, compare its retained snapshot to the matching
HDF5 integration (requires `h5py` and NumPy):

```sh
python3 monitoring/check_visibility_snapshot.py build/monitor-on/visibilities.hdf5
```

These replays establish correctness, not a live-throughput overhead
claim. Live on/off performance, wider geometries and the provisional <1%
overhead target require a calibrated observing session.

## Validation recorded 27 October 2026

- Built `starspin_1_8`; 8 WriterTests and 21 ProcessorTests passed.
- Five CPU monitor checks passed, including a continuously trickling partial
  Redis reply, snapshot ownership, bounded queue size and recovery.
- Full Python regression run: 67 passed, 5 skipped. The subsequently added
  trickled-reply test also passed independently.
- All 34 dashboard backend queries passed against Redis 8.2.2, Grafana 12.2.1
  and Redis datasource 2.2.0. Spectra and waterfall were also inspected in the UI.
- Disabled, enabled, absent-at-start and stopped/restarted Redis replay scenarios
  passed the independent visibility reference. The restart replay compared
  58,877,280 complex visibilities with maximum relative RMS below `9.36e-6`.
- After reconnect, all 22 published spectra matched their exact HDF5 integration;
  the Redis failure counter recorded two failures while correlation continued.

Local evidence is in `build_joss_clean/monitor-validation.json`,
`monitor-{disabled,enabled,unavailable,restart-final}/`,
`monitor-writers-test.log`, `monitor-processor-test.log`, and
`monitor-python-test.log` beneath that build directory. These artifacts are not
required for deployment. No live FPGA observing or overhead claim is included.
