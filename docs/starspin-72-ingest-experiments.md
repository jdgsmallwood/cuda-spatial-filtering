# 72-channel Starspin ingest experiments (2026-09-30)

Historical note (2026-10-02): the `starspin_4_72_timed*` trial target family was
removed from CMake after these runs. The commands and binary names below record
the experiments and are not current build instructions. The standard
`starspin_4_72` target and one `starspin_4_72_capture_only` benchmark remain.

The FPGAs were retuned to channels 160–231 on the already calibrated boot.
The same production `starspin_4_72` binary, delay file, CPU placement, and
120-second duration were used for the ring placement comparison. The capture
backend was libibverbs; receiver CPUs were 1,11,13,15, worker CPUs 3–6, and
the coordinator/GPU feeder CPUs 7–9.

| Receive ring NUMA nodes | Received | Missing | Missing fraction | Worst five-second increment |
| --- | ---: | ---: | ---: | ---: |
| NIC-local `0,1,1,1` | 498,925,547 | 1,153,840 | 0.231% | 155,795 |
| All node 0 `0,0,0,0` | 488,551,853 | 11,473,873 | 2.295% | 4,114,806 |

Both captures wrote 122 visibility blocks with increasing sequence numbers,
finite sampled values, and nonzero sampled values. The HDF5 files and console
logs are under `build/starspin_runs/ring_ab_{nic,node0}_72_20260930*`; the
console redirects are `/tmp/ring_ab_{nic,node0}_72_20260930_console.log`.
The node-0 layout has multiple severe loss bursts and should not become the
default. These are adjacent runs, so the exact loss rates may vary with system
load, but the magnitude and burst size strongly favor NIC-local allocation.

The launcher now accepts `--ring-numa-nodes LIST` for repeatable placement
tests; its default remains the NIC's sysfs NUMA node. Two matched experimental
executables were built:

- `build/apps/starspin_4_72_timed`: timing probes only.
- `build/apps/starspin_4_72_timed_opt`: the same probes plus one non-temporal
  store fence per worker batch and watermark publication for touched channels.

The timing probes report ring-full waits, worker batch time, completion pause,
completion lock wait/check, zeroing, handoff, buffer flight, and visibility
flush time. The direct-NIC baseline and optimized synthetic tests each passed
all three tests, including exact packet sample/scale values and completion.
Both experimental executables need `cap_net_raw+ep` after linking before a
live ibverbs A/B; the capability request is pending with the operator. Run the
same 72-channel, NIC-local, 120-second capture with each executable and compare
missing increments and timing lines before deciding whether to enable the
worker optimizations in production.

After the A/B, all four FPGAs were restored to channels 176–215 (40 channels,
center 196). A second tune invocation reported matching settings on all four
boards; see `/tmp/starspin_verify_restore_40_after_72_20260930.log`.

## Later 2026-09-30 timed session

The new experimental binaries received `cap_net_raw=ep`. At session start,
fresh eight-channel PCAPs showed nonzero samples and scale factor 1 on every
one of the 20 streams of every FPGA (12,000 packets per board). The initial
same-boot delay check failed to find a convincing shared peak on pair 0–2.
All four FPGAs were therefore rebooted (all DONE pins HIGH) and fully
initialized at channels 192–199 with ADC gain 20; all 20 ADC sync groups
passed. A new 30,000-packet-per-board capture solved all six pairwise delays
with **zero-sample closure error**. Its file is
`build/starspin_runs/alveo_delays_20260930_timed_ab_reboot.json`; its evidence
is under `build/starspin_runs/delay_capture_20260930_timed_ab_reboot/`.
Post-reboot PCAP samples remained nonzero, with scale factor 1 on all 100,000
scale entries inspected per board (5,000 packets each). No 64/128 scale
anomaly was found, so the FPGA `stats`, `r`, and `rawadcall` commands were not
needed. All four boards read back channels 160–231 before the live captures.

All following captures used the full Starspin GPU pipeline and the same
fresh delay file, NIC-local rings, and CPU placement:

| 120-second binary/run | HDF5 blocks | Missing | Missing fraction | Worst block | Worker tasks | Packets/task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| timed baseline | 122 | 2,726,646 | 0.545% | 6.88% | 2,171,343 | 229 |
| timed optimized, fence + touched channels | 122 | 7,786,206 | 1.546% | 27.93% | 17,459,211 | 27 |
| timed baseline repeat | 122 | 53,746,288 | 10.906% | 42.53% | 4,774,498 | 94 |

The baseline repeat shows severe run-to-run variation, so the first A/B does
not establish an optimization regression from missing fraction alone. It does
show that the fast-copy workers are consuming much smaller batches. An
additional `starspin_4_72_timed_opt_batched` target keeps both optimizations
and waits for up to 192 committed packets or 200 microseconds, whichever comes
first, before each worker task. Its three synthetic direct-NIC tests pass
exact packet sample and scale checks and completion. A 120-second full-pipeline
live run received the capability grant and produced 122 ordered HDF5 blocks
with finite, nonzero sampled visibilities. It missed 13,525,213 of
500,023,296 expected packets (2.706%); the worst block was 34.04% missing.
The worker task count was 2,459,925, or about 198 packets per task. Thus the
batch-size change worked, but severe loss bursts remained.

The strongest five-second loss interval in the batched run had 3,780,591
missing packets and 5,019 ms of accumulated ring-full wait across the four
receive threads, compared with roughly 1,000 ms in a quieter interval. The
repeat baseline had similarly large loss intervals with 7,500–8,500 ms of
ring-full wait. GPU buffer-flight maxima stayed about 34–39 ms across the
runs. This timing points to intermittent inability to drain the receive rings;
it does not identify whether the worker, completion coordination, or another
CPU-side collision initiates those stalls. The two worker optimizations should
remain experimental until a controlled longer comparison shows reliable loss
improvement.

After these captures, all four FPGAs were restored to channels 176–215. A
second tune invocation confirmed matching settings on each board in
`/tmp/starspin_verify_40_after_batched_20260930.log`.

## Completion interval and stall trace

`SPATIAL_COMPLETION_POLL_US` selects the sleep between completion checks.
The default is still 200 microseconds; the comparison will use 200 and 1000
with the same `starspin_4_72_timed_opt_trace` binary, optimized copying, and
192-packet batching. The exclusive completion lock remains in place. The
interval accepts 50–10000 microseconds and rejects invalid values at startup.

Set `SPATIAL_INGEST_TRACE_FILE` to an absolute CSV path to enable a bounded
trace with separate lanes for the coordinator, four workers, four receivers,
and GPU feeder. Memory is allocated and touched before capture starts; lanes
have one writer, recording requires no locks or I/O, and export happens only
after all trace producers join. Overwritten event counts are recorded. Every
completion cycle records wall time, lock-acquisition time, missing-packet
increment, and coordinator thread CPU time. Long worker pauses, batches,
receive-ring waits, receive progress gaps, and feeder submissions are recorded
on the same monotonic clock. A receive progress gap alone cannot distinguish
an idle input from descheduling; overlapping coordinator and ring-wait spans
and thread CPU time provide the next evidence.

Analyze an exported file with:

```bash
python scripts/benchmark/analyze_ingest_trace.py /absolute/path/trace.csv
```

`DIAG` and initial `ITER ibverbs` console reports are now opt-in via
`SPATIAL_VERBOSE_DIAGNOSTICS=1` in newly built binaries. Normal `Stats` reports
remain. Both interval trials must use the same quiet default because disabling
receive-loop diagnostic clocks and console writes may itself change throughput.

The colleague's shared launcher now defaults to the verified current boot's
`alveo_delays_20260930_timed_ab_reboot.json`, copied into its own run directory.
All four FPGA settings were checked again at 176–215 in
`/tmp/starspin_colleague_40_verify.log`. The system is being held at 40 while
the operator tests that launcher; compilation and further live experiments
wait for the operator to finish.


### Trace build and calibration verification, 2026-09-30

After the colleague finished the 40-channel launcher test, the tracing build
was completed. NVCC 12.8 crashed when the diagnostic flag used a lambda in a
member initializer alongside the timing probes. A plain helper function avoids
that compiler crash without changing the flag's meaning. The three direct-NIC
packet/completed-window equivalence tests pass with tracing enabled at both
200 and 1000 microseconds. The three trace/parser unit tests also pass.

A fresh eight-channel, 30,000-packet-per-board delay capture reproduced all
four existing offsets exactly, with zero-sample closure residual. Evidence is
in `build/starspin_runs/delay_capture_20260930_trace_verify/`. An additional
5,000 packets per board were inspected: every one of 100,000 scale entries
per board was 1, with nonzero samples (95.3–96.2% of scalar sample components).
No 64/128 anomaly was present. The existing boot-matched delay file remains
valid. No reboot was needed. All four boards read back channels 160–231 in
`/tmp/starspin_trace_verify72_20260930.log` before the live trace trial.


### Live scheduling and trace results (30-minute follow-up)

All captures below used the verified same-boot delays, NIC-local rings,
receiver CPUs 1/11/13/15, worker CPUs 3/4/5/6, and 72 channels 160–231.
The GPU runs include H2D, the full Starspin pipeline, D2H, and HDF5 output.
Only experimental binaries were used; production defaults were not changed.

| Run | Seconds | Pipeline CPU pool | Missing overall | Worst HDF5 block |
| --- | ---: | --- | ---: | ---: |
| `cpu_probe_72_20260930` (existing batched optimized binary) | 60 | 7–9 | 0.896% | 24.41% |
| `cpu78_probe_72_20260930` (same binary) | 60 | 7–8 | 0.806% | 17.02% |
| `isolated_coord_72_20260930` (**repeat baseline**, see below) | 120 | 7–9 | 2.596% | 38.19% |
| `cpu02789_probe_72_20260930` (same binary) | 120 | 0,2,7–9 | 2.130% | 32.91% |
| `trace200_72_20260930` (new quiet trace binary, 200 us) | 60 | 7–9 | 3.781% | 36.12% |
| `trace1000_72_20260930` (same quiet trace binary, 1000 us) | 60 | 7–9 | 6.755% | 29.12% |

All six HDF5 outputs have ordered block identifiers and finite, nonzero
sampled visibilities. Each one-minute run wrote 61 blocks; each two-minute
run wrote 122. These are short, variable runs, not evidence that a small
change in average loss is repeatable. The 1000-us interval did not solve the
problem; leave the default at 200 us. The old and new binaries differ in
verbose diagnostic behavior, so compare intervals within the new binary.

The attempted external `sched_setaffinity` operation on the running
capability-bearing process failed with EPERM on its first operation. No
thread affinity changed. Despite its filename, `isolated_coord` is therefore
a repeat baseline, not a successful isolation trial. Its sidecar
`.affinity_attempt.txt` records this explicitly.

The host-only `host_probe_72_20260930` run completed 6782 correlation input
buffers in 120 seconds, with 59,781 missing packets overall. After the first
five-second statistics report, missing increased by only 2940 over
479,179,696 received packets (about 0.00061%). This binary includes packet
unpacking and correlation-buffer creation, but no H2D or GPU processing.
It uses the existing host-only build, not the new optimized trace build, so
it is a diagnostic comparison rather than a perfectly matched GPU toggle.

#### What the trace establishes

In the 200-us run, all trace lanes retained every event (no overwritten
records). There were 160,275 completion gates but only 3391 GPU submissions.
The longest gate lasted **26.734 ms**: **26.729 ms before acquiring the
completion lock**, then 0.005 ms holding it. The coordinator consumed only
**0.011 ms of thread CPU time** in that entire interval. The longest worker
batch in the run was 0.994 ms. Receive-ring waits of about 26.5 ms on all four
NICs overlap this gate almost completely; for example, receiver 0's
26.547-ms wait overlaps completion gates for 26.399 ms. The longest gates
usually had no missing-packet increment and held the lock for only a few
microseconds, consistent with a check that had no buffer to hand over; the
missing increment alone does not prove whether a handover occurred. Other
gates do include millisecond-scale missing-data cleanup.

This directly ties the longest receive stalls to the global completion gate.
It shows the coordinator is off CPU or blocked for almost all of the long
lock-acquisition interval, not spending that time doing completion work.
It does not alone separate time asleep on the shared-mutex futex from time
runnable but waiting to be scheduled. Independent 1-ms `/proc/.../schedstat`
sampling in the older full-pipeline run found a 12.904-ms coordinator
run-queue wait with only 0.013 ms of CPU time in that sample; a 20-second
host-only sample found no run-queue increments above 1 ms.

CPU 9's earlier high NET_RX softirq load disappears during active ibverbs
capture: it was not saturated by softirq in these trials. Ancestor cgroups
have unlimited CPU quota and zero throttling. The H2D buffer reclaimer is a
plausible source of competing CPU work: it consumed about 90% of a core in
the same 7–9 pool. This observation does not establish it as the sole cause.

The next experiments should isolate the coordinator on CPU 7 while launching
other pipeline threads on 8–9, and avoid entering the global gate for checks
that cannot complete a buffer. A bounded spinning lock-acquisition experiment
may also distinguish futex wakeup latency from packet-copy time; this has not
been implemented. Preserve correctness around future-packet draining and
host-buffer recycling before changing the completion protocol.

`starspin_4_72_timed_opt_trace_isolated` adds opt-in
`SPATIAL_COORDINATOR_CPU=7`. With `--pipeline-cpus 8-9`, receiver/worker
explicit affinities remain as configured, and the coordinator sets its own
affinity before starting its workers. Affinity failure stops the run instead
of silently claiming successful isolation. The original trace executable is
preserved; the isolation executable still needs its own `cap_net_raw=ep` and
has not had a live FPGA trial.

Results and sampler scripts are saved under
`build/starspin_runs/trace_session_20260930/`. Event traces and human-readable
analyses are `build/starspin_runs/trace{200,1000}_72_20260930.{csv,analysis.txt}`.
Compilation was paused during both live interval trials and resumed only
after capture finished. At the end of the live window the FPGAs were restored
to 40 channels, 176–215; see `/tmp/starspin_trace_restore40_20260930.log` and
the independent readback `/tmp/starspin_trace_verify_restore40_20260930.log`.

The isolation build completed successfully. All three direct-NIC equivalence
tests pass at both 200 and 1000 us with the process launched on CPUs 8–9,
coordinator explicitly pinned to 7, and workers pinned to 3/4/5/6. All four
parser/trace unit tests pass. This is synthetic validation, not a live
throughput result for the isolated coordinator.

### Per-buffer completion decoupling (2026-09-30)

Checkpoint before this change: parent `33c2f8c`, TCC submodule `d71f1b4`.
The experiment is selected by `SPATIAL_DECOUPLED_COMPLETION` and built as
`starspin_4_72_timed_opt_trace_decoupled`. The production launcher still
selects its usual binary unless `--binary` is supplied.

`include/spatial/decoupled_ingest.hpp` replaces the global worker gate with
a generation and ownership state for each physical host buffer:

1. **Open:** each NIC worker writes its FPGA's samples, scales, and arrivals.
   Logical packet windows are calculated from an immutable, delay-corrected
   origin, rather than a shared changing `current_buffer`.
2. **Closing:** the coordinator requests closure after the existing
   all-channel/all-FPGA watermark threshold. Workers finish their current
   batch, fence streaming stores, and acknowledge that buffer's generation.
   Each worker continues processing other buffers. Idle workers also acknowledge.
3. **In flight:** after every acknowledgement, the coordinator masks missing
   samples with zero scales and queues the buffer to the GPU feeder. Workers
   cannot write the sealed generation.
4. **Recycled:** the H2D completion callback clears arrival flags and publishes
   the same physical slot for logical window `old_window + host_buffer_count`.
   Release order cannot change its logical window assignment.

Each worker owns its deferred-packet heap and receive-ring release decisions.
Packets spanning adjacent windows remain queued until both destinations exist,
preserving overlap samples during H2D backpressure. The bounded-skew timeout
may advance watermarks on a stream jump, but cannot acknowledge a closing
window while that worker still owns queued packets for it. Genuine exhaustion
of all host buffers still causes backpressure; decoupling does not create
unlimited storage or fix an independently slow GPU.

The experiment preserves the 1536 slots per NIC, up-to-256-packet work batches,
192-packet/200-us accumulation policy, huge-page input allocation, and existing
affinity controls. It supports live CPU-ring capture with one FPGA per worker;
PCAP replay and GPUDirect capture are rejected by this experimental executable.
Trace event 8 records request-to-acknowledgement time per buffer; this is not
a global worker pause. Worker-copy and receive-wait timing remain available.

`ProcessorDecoupledTests` checks exact sample/scale/overlap data, packet-aligned
FPGA delays, intentional missing packets, stale writes after seal, a stalled
completion callback, repeated slot reuse, out-of-order H2D releases, forced
watermarks, large stream jumps, and shutdown with deferred packets. Its GPU
test uses the asynchronous feeder and real H2D release callback, executes the
Starspin GPU pipeline, and compares every visibility component against an
independent CPU correlation (relative tolerance 2e-5, absolute floor 0.01).
That small numerical test uses one fine channel; the production 72-channel
configuration and its fine channelizer still require a live A/B run.

Validation result: all eight tests passed 20 consecutive repetitions with
the process on CPUs 7–9 and workers on 3/4/5/6 (160 test executions).
The experimental application built successfully and `run_starspin.sh --dry-run`
selected CPU-ring ibverbs capture and NUMA ring nodes `0,1,1,1` as expected.
Logs: `/tmp/starspin_decoupled_final_build.log`,
`/tmp/starspin_decoupled_repeat.log`, `/tmp/starspin_decoupled_dry_run.log`.
No FPGA capture or tuning was performed during this implementation turn.

Build and synthetic validation (no FPGA tuning or capture):

```bash
cmake -S . -B build
cmake --build build --target ProcessorDecoupledTests \
  starspin_4_72_timed_opt_trace_decoupled -j4
env CUDA_HOME=/data/BLACKMESA_1/sma075/miniconda3/targets/x86_64-linux \
  SPATIAL_WORKER_CPUS=3,4,5,6 build/tests/ProcessorDecoupledTests
```

Before live use, grant the rebuilt executable its capability:

```bash
sudo setcap cap_net_raw=ep build/apps/starspin_4_72_timed_opt_trace_decoupled
```

For the next authorized live session, first verify FPGA boot/delays and retune
to 72 channels as required by `AGENTS.md`. Then compare this command with the
baseline `starspin_4_72_timed_opt_trace` under the same conditions:

```bash
SPATIAL_INGEST_TRACE_FILE="$PWD/build/starspin_runs/decoupled72.csv" \
scripts/run_starspin.sh --channels 72 --min-channel 176 \
  --binary "$PWD/build/apps/starspin_4_72_timed_opt_trace_decoupled" \
  --delay-file /path/to/verified-current-boot-delays.json \
  --pipeline-cpus 7-9 --duration 300 --name decoupled72
```

Leave `SPATIAL_COORDINATOR_CPU` unset for the first matched comparison; test
isolation separately. No live throughput or packet-loss improvement is claimed
from the synthetic tests.

#### First live decoupled run and system release

`decoupled72_full_20260930` completed 300 seconds at channels 160–231,
including CPU-ring ibverbs capture, packet assembly, H2D, the production fine
channelizer and correlator, D2H, and HDF5 writing. It wrote 303 visibility
blocks with continuous sequence ranges. HDF5 metadata records 5,852,801
missing packets out of 1,250,058,240 expected: **0.468202%** overall,
**8.209592%** in the worst block, and 29 blocks above 1% missing. Excluding
the first block gives 0.466159%, so the remaining loss is not just startup.
No matched baseline was run: the user requested system release after this run.
These figures therefore do not establish a controlled A/B improvement.

The final timing counters show no global completion pause/lock wait; the
maximum receive-ring wait was 3.029 ms and the maximum worker copy batch was
0.682 ms. Buffer flight reached 38.525 ms; no deferred packets or stuck
unprocessed packets were reported. GPU telemetry showed Gen 3 x16 in every
active sample. The application shut down cleanly and both monitoring processes
were stopped. Only small HDF5 metadata datasets and saved logs were read for
the final summary; a full visibility-array validation was not performed after
the stop request. Artifacts are `build/starspin_runs/decoupled72_full_20260930.*`.

Session setup included a fresh four-board reboot (FPGA 3 needed a retry),
full eight-channel initialization, and six-pair delay calibration. The scale
anomaly recurred on FPGA 2 ADC stream 14. Before recovery, `stats`, `r`, and
`rawadcall` were run and all 20 raw ADC channels were saved. Reinitialization
alone did not clear it; a targeted FPGA 2 reboot did. Final eight-channel
captures had scale 1 in every inspected scale value on all four boards and
nonzero samples. See `/data/BLACKMESA_0/lambda/debug_pcaps/fpga2_scale_anomaly_20260930/README.md`.
The final delay solve had zero closure residual and is
`build/starspin_runs/alveo_delays_20260930_decoupled_final.json`. Both the local
and shared Lambda launchers now default to this file; earlier boot-state delay
files are historical. Readback confirmed 72 channels on all four boards.

**State at user-requested release:** no capture or monitoring remains active;
FPGAs remain tuned to **160–231 (72 channels)**. No further retuning or baseline
run was performed after the user's stop instruction.
