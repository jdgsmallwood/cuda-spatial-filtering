# Handoff: `--capture-backend ibverbs` zero-copy (`SPATIAL_IBVERBS_ZERO_COPY=1`)

Continuation entry point for `include/spatial/libibverbs.hpp`'s zero-copy receive path and the
shared buffer-placement code in `include/spatial/spatial.hpp` it exposed bugs in. Written
2026-09-20 (multiple real-time gaps during this session — check file mtimes, not just the
narrative order, if timeline matters). This is a **separate line of work** from the
"mailbox"/H2D-benchmark investigation in `docs/ingest-optimization-handoff.md` — don't merge
the two; they touch overlapping files but are answering different questions.

## 2026-09-21 current state: runtime re-seeding removed; 32ch proven live

**40-channel update:** 40-channel support was added to the CMake target grid and
`build/apps/starweave_4_40` built successfully with four FPGA sources. A live 60-second ibverbs
zero-copy smoke test covered channels 176–215 (`--min_freq_channel 176`) and completed cleanly:

```text
Received=138815720, Processed=138814912, Missing=73522, Discarded=11913,
FutureQueued=0, StuckUnprocessed=0, NICDrops=0
Pipeline Runs Queued = 3390
```

This sustained **2.313595 Mpps**, or approximately **6.154 GB/s** at 2,660 bytes per packet.
Total missing was **0.052935705%**, including startup. Most missing accrued in the first 30
seconds; the counter then stayed flat for five reports before a final 394-packet increment in the
60-second report. Initialization occurred exactly once and startup discards stayed flat. The
808-packet Received/Processed difference is the shutdown tail. Artifacts use the prefix
`build/starweave_smoke_out/ibverbs_40ch_oneshot_60s`.

**48-channel investigation:** the target grid now also includes 48 channels and the centered test
range is 172–219 (`--min_freq_channel 172`). The first 60-second smoke test froze after
11,383,508 received / 11,381,456 processed, with `FutureQueued=513`, `StuckUnprocessed=48`, and
233 pipeline runs. Worker diagnostics identified ring slot 3908 as reserved but never committed.

A first attempted fix retired the entire CQ batch before reserving replacements. It still froze,
this time after 16,862,793 packets with four worker lanes simultaneously waiting on reserved but
uncommitted slots. That attempt was reverted; the exact source behavior proven at 40 channels is
preserved in commit `4413a83`.

The current 48-only experiment leaves the <=40-channel path at its proven fixed dispatch size of
256, but selects a compile-time fixed size of 64 for targets with 48 or more channels, matching
the ibverbs CQ poll batch. This is not adaptive receive batching. Rebuild and smoke validation are
pending; if it passes, proceed to the operator-requested 600-second run.

This section supersedes the older recommendations below about a timed re-seed window or a
collection-based bootstrap. Those sections are retained as investigation history only.

The processor now uses the original/simple startup contract the operator expects:

1. The first valid packet to acquire `buffer_bootstrap_mutex` calls
   `initialize_buffers(sample_count, fpga_id)` exactly once.
2. `initialize_buffers()` uses the configured delay file to translate that packet's counter into
   the corresponding starting counter for every FPGA.
3. `buffer_init_flag` publishes those boundaries and they are never changed during the run.
4. Packets older than that chosen startup boundary can be discarded. A short startup loss is
   acceptable; the requirement is convergence to loss-free steady state within about one second.

There is no remaining runtime re-seed/CAS path, no timed re-seed window, and no requirement to
wait for one packet from every FPGA. This avoids repeatedly destroying partially-filled buffers
when startup packets from the four capture threads are interleaved. Delay arithmetic in
`initialize_buffers()` is now explicitly signed, including the live FPGA-2 delay near -80.8
billion samples, and rejects a genuinely negative resulting raw sequence instead of allowing
unsigned wraparound.

Verification completed after the change:

- `ProcessorStateHugeDelayBootstrapTest.FirstPacketInitializesOnceAndLaterStreamsConverge`
  exercises four sources with delays `{8800000, 0, -80800000000, -16800000}`. Earlier packets
  from later-arriving sources are discarded, later packets from all four sources are placed, and
  the initial buffer boundaries remain unchanged.
- Full `build/tests/ProcessorTests`: **18/18 passed**, including both variants of
  `FourFpgaHandoffTest` (legacy and mailbox).
- `starweave_4_32` rebuilt successfully and passed a live 60-second ibverbs zero-copy run at
  `--min_freq_channel 180` after restoring `cap_net_raw`.

The live validation passed the stated convergence requirement. Final counters were:

```text
Received=111037786, Processed=111037568, Missing=67334, Discarded=4640,
FutureQueued=0, StuckUnprocessed=0, NICDrops=0
Pipeline Runs Queued = 3390
```

This is 1.8506 Mpps received (about 4.923 GB/s at 2,660 bytes per packet). The 218-packet
Received/Processed difference is the shutdown tail. `Discarded` reached 4,640 before the first
statistics report and remained exactly 4,640 for all 12 reports. `Missing` reached 67,334 in the
early reports and then remained exactly flat for the final eight reports. Buffer initialization
occurred exactly once, from FPGA 0 counter `103853549824`; all four streams then advanced, all
3,390 pipeline runs were queued, and shutdown completed normally. This demonstrates that the
former 32-channel freeze and sustained-discard failure are fixed.

Artifacts:

- `build/starweave_smoke_out/ibverbs_32ch_oneshot_60s.log`
- `build/starweave_smoke_out/ibverbs_32ch_oneshot_60s.hdf5` (2.2 GiB)
- `build/apps/app.log` (one-shot initialization and buffer-boundary evidence; remember that the
  next process launch truncates this file)

Rate interpretation: 1.8506 Mpps is 93.3% of the nominal 1.984 Mpps objective, but the application
kept up with all traffic it saw: zero `NICDrops`, zero stuck work, and only a 218-packet shutdown
tail. Most importantly for the operator's acceptance criterion, `Missing` did not grow during the
final roughly 40 seconds. A separate 10.0046-second NIC measurement with Starweave stopped found
1.8582 Mpps offered across the four interfaces (`rx_packets + rx_missed_errors`), closely matching
the application's 1.8506 Mpps capture rate. Thus the current gap from the nominal 1.984 Mpps is
in the live offered traffic, not evidence that this ingest path is falling behind. Interfaces
94 and 134 overflow their hardware rings while no application drains them; that is expected for
this stopped-application measurement and is why offered rate uses delivered plus missed packets.

The saved run-specific NIC "before" snapshot was taken before the 4m44s NVRTC startup, while FPGA
traffic continued, so that particular before/after delta spans much more than the 60-second capture
and must not be used as a capture-window loss measurement.

### Ten-minute confirmation

A subsequent 600-second run of the identical 32-channel configuration also completed and shut
down cleanly:

```text
Received=1111114882, Processed=1111113728, Missing=32461, Discarded=11552,
FutureQueued=0, StuckUnprocessed=0, NICDrops=0
Pipeline Runs Queued = 33908
```

Using `Missing / (Received + Missing)`, the total missing fraction is **0.002921395%**. This
includes startup. Missing-counter changes were sparse: 18,988 were already present in the first
five-second report; the final increment was only 138 packets at approximately 480 seconds, after
which `Missing=32461` remained flat for the final two minutes. Buffer initialization occurred
exactly once, startup `Discarded=11552` never grew after the first report, and the 1,154-packet
Received/Processed difference is the shutdown tail. Sustained receive rate was 1.851858 Mpps.

Artifacts:

- `build/starweave_smoke_out/ibverbs_32ch_oneshot_600s.log`
- `build/starweave_smoke_out/ibverbs_32ch_oneshot_600s.hdf5` (about 16 GiB)
- `build/starweave_smoke_out/ibverbs_32ch_oneshot_600s.streams.csv`

## Historical state before the 2026-09-21 one-shot change

**2026-09-20, later update**: the buffer re-seed gate went through a third iteration after the
`any_buffer_completed` latch (fix #4's "current fix" as originally written) was found to regress
the previously-clean 24ch kernel-backend small-delay case. Replaced with a wall-clock-bounded
window (`first_init_time` + 2s) — see fix #4 below for the full story and verification status.
This was found/fixed with no `cap_net_raw` available, so only kernel-backend re-validation was
possible this pass; ibverbs zero-copy re-validation (including whether this also helps or still
needs more work for the 32ch/FPGA-restart open issue) is still pending a fresh capability grant.

**2026-09-20, even later update — important methodology correction**: the "regression" that
motivated the above rewrite is now believed to be **partly a stale-test artifact, not purely a
code regression**. FPGA 2's restart (giving it a ~-80.8 billion relative delay) is **live,
permanent hardware state as of this session** — it's baked into every real packet's
`sample_count` on the wire now, not just a value in a test `alveo_delays.json`. The "isolated
regression check" that first caught the problem fed a **stale, artificially small** delay value
for FPGA 2 (`-533348679`, the pre-restart value) against **current, post-restart live traffic**
— i.e. it tested a genuinely wrong delay-to-hardware mapping, which will always misalign buffers
badly regardless of any re-seed gate logic, since `initialize_buffers()`'s own delay math has no
way to know the configured delay doesn't match reality. That run's `Discarded=1,030,971`,
`StuckUnprocessed=61,338` result is real, but its cause is more subtle than "the latch is too
loose" — see the gdb-confirmed root cause below. **Live capture is not a reliable way to
regression-test buffer-init-race fixes anymore, now that FPGA 2's delay has permanently changed**
— a PCAP replay captured before the restart (deterministic, decoupled from current hardware
state) would be the correct way to re-validate the small-delay case in a future session.

Separately, and confirmed independently via `gdb -p <pid> -batch -ex "thread apply all bt"` on a
live, stalled process: **running the actual current scenario correctly** (32ch,
`--min_freq_channel 180`, the real current `alveo_delays.json` with FPGA 2's genuine
-80.8-billion delay) still deadlocks under the kernel backend, with all 4 capture threads
permanently parked inside `reserve_write_batch_strided()` (called from
`KernelSocketPacketCapture::get_packets()`) and the processor thread parked inside
`process_packets()` — i.e. the packet ring (`NR_INPUT_BUFFERS=8` × `RING_BUFFER_SIZE=6144`) is
completely full of buffers that can never complete correlation, so nothing ever frees a slot for
capture to reserve into. Live NIC `rx_packets` counters were checked before/during/after and
confirmed real traffic kept flowing throughout (ruling out "FPGA stopped transmitting" as the
cause) — this is a genuine application-level convergence failure, not a hardware/wiring issue.
**This matches the pre-existing, already-documented "Open issue" below almost exactly** (reactive
single-packet re-seeding doesn't converge to the true per-FPGA minimum before buffers lock in)
— it is very likely the *same* unresolved problem, not a new one introduced by the wall-clock gate
rewrite, though the shorter 2s window (vs. the latch's much longer effective window) plausibly
makes it converge *even less* often than before. **Bottom line: neither gate variant (latch or
wall-clock window) actually solves 32ch+huge-delay convergence** — both are reactive,
single-packet-at-a-time approaches with the same fundamental limitation described in "Open issue."
The wall-clock version is kept as the current code (it's a strict improvement for the *intended*
startup-race scenario — short, throughput-independent, doesn't stay open indefinitely — and is no
worse than the latch for the large-delay case), but the real fix for 32ch+huge-delay is still the
collection-based redesign described below, not yet attempted.

**24-channel (`starweave_4_24`, `--min_freq_channel 184`, no unusually large `alveo_delays.json`
entries) is fully fixed and proven**: 83.33M packets received/processed (~1:1) over a clean 60s
run, `Missing` flat at ~0.0095%, zero timeouts, `Pipeline Runs Queued` actually incrementing.
~1.39M pps sustained — substantially higher than the kernel-socket backend's throughput this
session. See "Fixes landed" below for what got it there.

**32-channel with one FPGA carrying a very large relative delay (a live example: FPGA 2 restarted,
`alveo_delays.json` delay `-80801570823` vs `8838643` / `0` / `-16809804` for the other three) is
NOT yet reliable** — see "Open issue" below. This is the active problem as of handoff.

## Fixes landed (all in this session, all confirmed necessary via direct A/B testing)

1. **Ring-index bug** (not mine — a concurrent session/user fixed this first): `arm_zero_copy()`'s
   strided cursor must start at `thread_id_`, not 0, or all four capture threads' QPs claim the
   same ring slots and stall. `include/spatial/libibverbs.hpp` ~line 405.

2. **Header-offset corruption** (real, independent bug, mine): `post_direct_recv()` posted a
   single-SGE receive WR landing the *entire raw Ethernet+IP+UDP frame* at the ring slot's
   `->data[0]`, where every downstream packet parser expects **payload-only** (matching what
   `KernelSocketPacketCapture`'s `recvmmsg` over a `SOCK_DGRAM` socket delivers there — the kernel
   already strips L2/L3/L4 headers for that path). This file's own other two receive paths
   (`post_recv`'s `separate_header=true` branch, and `LibibverbsGpuDirectPacketCapture`'s 3-SGE
   setup) already handle this correctly with a discard SGE for the header; the zero-copy path was
   missing it. Fixed with a **per-frame** header landing zone (`header_ring_zc_`, indexed by
   `jframe`) — **do not use a single shared discard buffer**, that races when a poll batch
   contains more than one completion (whichever's DMA lands last "wins" the buffer, not
   necessarily the one currently being processed). `wc[i].byte_len` also had to be corrected
   (`payload_len = len - TOTAL_HDR_SIZE` for zero-copy, since `byte_len` is the *total* across
   both SGEs) for both the committed packet length and the IP-header source-address recovery
   (which now reads from `header_ring_zc_[jframe]`, not from `frame`, since `frame` is
   payload-only post-fix).

3. **Bounded WR-retry timeout**: the original code's replacement-WR reservation retry
   (`reserve_write_batch_strided` failing on ring-full) looped forever with no diagnostic. Added a
   3s wall-clock timeout with a `std::cerr` log (bypasses the shared `spatial::Logger` singleton
   deliberately — see the logging pitfalls section below) before giving up on that one WR. This
   didn't fix the underlying stall by itself but made it *diagnosable* instead of a silent hang —
   directly led to finding fix #5.

4. **Buffer-init race** (real bug in shared `spatial.hpp`, not ibverbs-specific, but ibverbs'
   massive near-simultaneous initial burst — ~1024 pre-posted receives completing within
   microseconds of each other at startup — makes it far more likely to trigger than the
   kernel-socket path's naturally-paced delivery): `initialize_buffers()` seeds every buffer's
   `start_seq` from whichever packet's worker thread happens to win `buffer_init_flag`'s mutex
   race, not necessarily the packet with the genuinely lowest `sample_count`. A later-sequenced
   packet winning means every genuinely-earlier packet looks "impossibly early" and gets
   discarded forever (`packet_index_for_buffer(...) < -1` at spatial.hpp's
   `process_packet_data()`). Fix: when this discard condition fires, re-seed via
   `initialize_buffers(pkt.sample_count, pkt.fpga_id)` (reusing its own already-correct,
   delay-aware math) instead of discarding — but **gated correctly** matters a lot, see the two
   wrong attempts below before landing on the right one:
   - **Wrong attempt 1**: gate on `packets_processed.load() < 64` (a packet-count threshold).
     Broke once fix #6 (below) let the pipeline actually reach full throughput: at tens of
     millions of packets over a few seconds, *any* fixed count closes long before a large,
     genuine cross-FPGA misalignment finishes settling, so correct-but-late packets after that
     point still got hard-discarded. This is exactly the 32-channel/FPGA-restart open issue.
   - **Wrong attempt 2**: gate on `buffers[buffer_index].is_ready`. Also wrong, differently:
     `is_ready` legitimately cycles `true`/`false`/`true` throughout **normal steady-state
     operation**, once per buffer recycle — it doesn't distinguish "still starting up" from
     "buffer 0 just happens to be between dispatches right now". Using it would let re-seeding
     fire on an ordinary late/reordered packet well into a long-running observation, destructively
     resetting every buffer's `start_seq`/`end_seq` and corrupting real, already-accumulated data.
   - **Wrong attempt 3**: a one-time latch, `any_buffer_completed` (`std::atomic<bool>`, set `true`
     exactly once when the first buffer genuinely completes). Fixed the 32ch/FPGA-restart discard
     rate somewhat (re-seed fires, confirmed via its log line, 24 times in one 30s run) but was
     later discovered to be a **regression on the previously rock-solid 24ch kernel-backend
     small-delay case**: the latch stays permissive until the *entire first buffer* completes,
     which at real throughput can be many seconds — long enough for ordinary packet
     jitter/reordering during that window to trigger repeated destructive full-buffer re-seeds.
     Isolated via a clean A/B (same binary, original small `alveo_delays.json` values, kernel
     backend only): `Discarded=1,030,971`, `StuckUnprocessed=61,338` — a frozen, badly broken run,
     versus the established zero-discard baseline. Removed entirely (including its `.store()` at
     buffer completion) rather than kept alongside the fix below, since it had no remaining reader.
   - **Current fix**: a short **wall-clock-bounded window** from the initial `initialize_buffers()`
     call, not tied to buffer completion or packet count at all. Added
     `std::chrono::steady_clock::time_point first_init_time` (`spatial.hpp`, near
     `buffer_init_flag`), written once right before `buffer_init_flag`'s release-store inside the
     existing double-checked-lock block (the release/acquire pairing on `buffer_init_flag` already
     makes this write visible to any thread that observes the flag as true, so no extra
     synchronization was needed). The re-seed gate at the discard-check site now reads:
     `buffer_init_flag.load(acquire) && (steady_clock::now() - first_init_time) < 2s`. Two seconds
     is comfortably longer than any genuine startup-burst settling time (observed sub-millisecond
     to low-milliseconds for ibverbs' ~1024-receive initial burst) while being clearly shorter than
     any realistic first-buffer fill time at real observation throughput, so it can't overlap with
     ordinary steady-state jitter the way the latch did. **Verified**: rebuilt `starweave_4_24` and
     re-ran the same isolated kernel-backend/small-delay regression check that caught the latch
     regression — see the result noted at the top of this section once confirmed. Not yet
     re-validated against the 32ch/FPGA-restart ibverbs scenario (needs `cap_net_raw`, unavailable
     this handoff — see the constraints section below); a 2s window may or may not be generous
     enough for that case's genuine misalignment to fully settle, unlike the latch it replaced
     which had no such ceiling. If 32ch/FPGA-restart testing resumes and still shows heavy discards
     within this fix, that's the first thing to suspect — either widen the window or move to the
     collection-based approach described in "Open issue" below.

5. **`packet_index_for_buffer` signed-underflow bug** (real, independent, severe bug in shared
   `spatial.hpp`, found while investigating the 32-channel discards): computed
   `(sample_count - buffer_start)` in **unsigned** 64-bit arithmetic. Whenever `sample_count <
   buffer_start` — which is the *normal*, expected case this function exists to detect, for the
   `packet_index < -1` discard/defer check to work at all — the subtraction underflows and wraps
   to a huge positive `uint64_t`, which the function's `int` return type then truncates
   essentially arbitrarily (sometimes staying visibly out of range, sometimes wrapping back into
   what looks like a small, valid in-buffer index — silent data corruption, not just wrong
   discards). Latent but rarely triggered for the tens-of-millions-scale delays this path was
   first proven against; became severe with an FPGA-restart-scale delay (~-80.8 billion). Fixed:
   do the subtraction in signed `int64_t` (both operands always comfortably fit — sample counts
   don't approach `int64_t`'s ~9.2e18 range), then clamp to ±1,000,000,000 (far beyond any real
   buffer size) before the final narrowing cast, so a legitimately huge mismatch clamps to an
   unambiguous "very out of range" sentinel instead of wrapping. **This fix is verified correct in
   isolation** (discard log messages now show sane, small begin_seq/sample_count gaps — thousands,
   not the billions-scale garbage seen before) but the 32ch/FPGA-restart case still discards
   heavily for the reason in fix #4's "current fix" above, not because of a recurrence of this bug.

6. **The actual root cause of the "Received permanently freezes" deadlock** (the deepest, most
   consequential fix): `process_packets()`'s dispatch loop (`spatial.hpp`, the `for (tid...)`
   strided path) hands one capture thread's *entire currently-claimed span* to
   `dispatch_and_wait()` in a single blocking call, sequentially per thread — capped by
   `REGULAR_BATCH_SIZE`, which was `6000`. `dispatch_and_wait()` blocks until every slot in its
   batch is committed, and `per_thread_read_linear[tid]` (hence the `Stats:` line's "Processed"
   counter) only advances **once the entire call returns** — not incrementally. For
   `KernelSocketPacketCapture`, reservation happens synchronously with the `recvmmsg` batch that
   fills it (capped at 256 packets), so this was essentially never the limiting factor. For
   zero-copy ibverbs, the receiver commits one packet and immediately reserves+posts a replacement
   continuously, independent of how far that puts it ahead of actual processing (bounded only by
   `reserve_write_batch_strided`'s own ring-full check, the *entire* `RING_BUFFER_SIZE`) — so a
   single thread's claim can run thousands of slots ahead of what's been processed, and the
   processor gets stuck for the *entire test duration* inside one `dispatch_and_wait()` call for
   that one thread's oversized, mostly-uncommitted backlog, while the other three threads'
   already-ready data goes completely untouched. Confirmed by direct measurement: more workers
   (3→6) barely moved `Processed` (384→440) because the bottleneck wasn't worker parallelism, it
   was how work got handed out in the first place.
   **A throttle-based fix (bounding the receiver's reservation lookahead) was tried first and
   is wrong** — don't reintroduce it. It doesn't address the batch-size mechanism at all; it just
   starves the receiver of new data to feed the *same* still-oversized, still-incomplete batch,
   making things measurably worse (`Processed` went to 0 and stayed there for the whole test).
   **The actual fix**: reduce `REGULAR_BATCH_SIZE` from `6000` to `256` (`spatial.hpp`, matches
   `KernelSocketPacketCapture`'s own already-tuned `BATCH_SIZE`). This forces every
   `dispatch_and_wait()` call to be small and fast regardless of how far any one thread's claim
   has raced ahead, restoring the round-robin fairness the surrounding code's own comment already
   describes as the design intent ("Process each thread's claimed range independently so a slow or
   idle thread never stalls the others"). **Verified as a no-op for the kernel-socket path**
   (regression-tested at 24ch kernel backend post-fix: `NICDrops=0`, `StuckUnprocessed=0`,
   `Discarded=0`, matching the established known-good pattern exactly) since its batches never
   approached the old 6000 cap in the first place. This single change is what took the 24-channel
   case from a hard 2052-packet ceiling to 83M+ packets sustained.

## Open issue: 32-channel with a large FPGA-restart-scale delay (as of handoff)

Symptom: heavy, sustained discarding (tens of thousands of packets, up to ~50% of received) and
eventually the same ring-fill deadlock/TIMEOUT pattern fix #6 was supposed to have eliminated,
specific to this scenario (`alveo_delays.json` with one FPGA's delay several orders of magnitude
larger than the others, from an FPGA restart resetting its packet counter).

Root cause (confirmed, not yet fully fixed): the re-seed logic in fix #4 is a **reactive,
single-packet-at-a-time** correction — each "too early" packet individually triggers a full
re-seed using *that one packet's* `sample_count`, with no coordination across the (potentially
many) other packets from other threads/FPGAs that are also "too early" around the same time. Under
a large, genuine misalignment, many such packets arrive in quick succession before the true
minimum has been found, and each re-seed simply overwrites the previous one — there's no guarantee
the process converges to the actual minimum before `any_buffer_completed` latches true (a buffer
completing with boundaries that are still wrong, then locking in that mistake for good). Confirmed
via log evidence: the re-seed line fires (24 times in one 30s test) but ~74,625 packets still hit
the hard-discard path in the same run.

**Confirmed still reproducing under the kernel backend too** (2026-09-20, later): this isn't
ibverbs-specific — the same convergence failure was independently reproduced via
`--capture-backend kernel` (no `cap_net_raw` needed) against the real current `alveo_delays.json`,
and root-caused via `gdb -p <pid> -batch -ex "thread apply all bt"` on the live stalled process:
all 4 capture threads permanently parked in `reserve_write_batch_strided()`
(`KernelSocketPacketCapture::get_packets()`), processor thread parked in `process_packets()` —
the ring is full of buffers that never complete, so nothing ever frees a slot. Live NIC
`rx_packets` confirmed via before/after snapshots to keep climbing throughout (real traffic, not a
stalled FPGA) — this rules out a hardware-side explanation. See the top-of-file 2026-09-20 update
for the full writeup. Useful for next time: **`gdb -p <pid> -batch -ex "thread apply all bt"`
against a live, stalled process needs no `sudo`/`cap_net_raw`** (gdb attach to your own-uid
process is unprivileged) and is a much faster way to distinguish "ring deadlock" from "socket/NIC
problem" than staring at `Stats:` counters — use it early next time instead of guessing.

**Recommended fix for the next session** (not yet attempted — needs a fresh `cap_net_raw` grant
via `sudo setcap cap_net_raw+ep`, unavailable at this handoff): replace the reactive
one-packet-at-a-time re-seed with a **collection-based** approach — before committing to any
buffer initialization, gather the observed minimum `sample_count` *per FPGA source* over a short
initial window (either a fixed small packet count per source, or a short wall-clock window, e.g.
50-100ms), THEN call `initialize_buffers()` once with the genuine minimums, rather than reactively
re-seeding on every individual "too early" event. This avoids the "last writer wins, not lowest
writer wins" race entirely. Needs care around: (a) not blocking real processing indefinitely if
one FPGA's first packet never arrives (timeout/fallback), (b) whether the "per FPGA source"
minimum needs to also account for the already-correct delay math in `initialize_buffers` (probably
yes — collect raw `sample_count` per source, feed the *lowest* source's raw value through the same
`first_count`/`fpga_id` single-value entry point `initialize_buffers` already has, since its own
delay-alignment math handles converting one source's raw count into all buffers'/sources'
`start_seq`/`end_seq` correctly already).

## `cap_net_raw` / privilege constraints (important for any continuation)

- `starweave_4_{16,24,32}` need `cap_net_raw` for the RAW_PACKET ibverbs QP (also documented in
  the top-level `CMakeLists.txt`'s comment about the same requirement for `ibverbs_pcap_sender`).
- **Every rebuild strips the capability** (it's a file xattr, not preserved through recompilation)
  — `sudo setcap cap_net_raw+ep <path-to-binary>` is needed after *every single* `cmake --build`
  of an ibverbs-capable target, not just once.
- At this handoff, no further `sudo` access is available in this session. Any further ibverbs
  zero-copy testing needs a fresh grant from whoever has it. The kernel-socket backend
  (`--capture-backend kernel`) does *not* need this and remains fully testable without root.
- `LD_PRELOAD` does not survive alongside a `cap_net_raw` file capability (glibc's secure-exec
  mode strips it) — irrelevant to this specific ibverbs work but relevant if VMA
  (`docs/starweave-capture-runbook.md`'s experimental section) is revisited alongside it.

## Logging pitfalls hit this session (save future debugging time)

- `app.log` (the file `spatial::Logger`/`INFO_LOG`/`ERROR_LOG` write to, once `setup_logger()`
  redirects it — see `include/spatial/common.hpp`) is **truncated on every process start**
  (`basic_file_sink_mt(..., true)`). Check it *immediately* after the run you care about, before
  launching anything else — a later run's startup will silently erase the previous run's content.
- A background wrapper script's `until grep -q "Setup completed" <log>; do sleep 1; done` can spin
  forever even though the text is visibly present if you don't pass `grep -a` — some of these logs
  apparently trip `grep`'s binary-file heuristic (locale-dependent multi-byte sequence validation,
  most likely) without actually being binary per `file(1)`. This caused numerous false "the process
  never reaches setup" diagnoses this session that were actually just this. **Always use `grep -a`
  for these logs, everywhere, including inside detached background scripts** — a plain `grep -q`
  passed silently in every interactive check because those happened to use `-a` out of habit,
  while the *scripted* checks didn't, producing a long, confusing string of apparently-hung
  background jobs that were actually fine.
- `ibv_get_async_event()` on `ctx_->async_fd` **blocks by default** (not documented obviously
  in-context) — a diagnostic that calls it needs `fcntl(fd, F_SETFL, O_NONBLOCK)` set correctly
  first, or the diagnostic itself can hang the thread it's meant to be diagnosing. Tried and
  removed this session after it looked like it might be the cause of a run producing zero log
  output for its entire duration; the safer replacement (a plain wall-clock-gated counter dump via
  `std::cerr`, no syscalls beyond `ibv_poll_cq` itself) is what actually led to finding fix #6.
