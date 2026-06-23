# Testing strategy: the shared pipeline-test harness

## The problem this solves

Of the 7 `Lambda*Pipeline` GPU pipeline variants (see `include/spatial/pipeline.hpp`), only one
(`LambdaGPUPipeline`) had any test coverage, via `test_pipeline.cu`'s ~1,300-line `Ex1` and
friends. Most of that file is reusable setup boilerplate — building synthetic wire-format packets,
constructing the pipeline, wiring up output buffers, driving the buffer-fill/flush sequence,
syncing the GPU — repeated across its cases. Writing a test for any of the other 6 variants meant
copy-pasting most of it and editing the parts that differ.

`tests/support/` is the shared 80%: it turns "test pipeline variant N" into "construct it, call
`.run(...)`, assert invariants on the `Output`."

## Test at a high level, not against implementation details

The guiding principle: tests should encode **physical and mathematical invariants of the data**
(finite-ness, Hermitian symmetry where it's actually stored, eigenvalue ordering, "a known tone
shows up at the expected FFT bin", ...) rather than asserting on specific intermediate computation
paths or magic constants. That makes them survive refactors of the pipeline internals — including
the LOC-reduction refactor of `pipeline.hpp` this test-coverage push exists to enable — instead of
breaking in lockstep with them.

The one deliberate exception: where the math is simple enough to hand-derive exact expected
values (as `Ex1` does for `LambdaGPUPipeline` with unity weights and constant input), exact-value
golden checks are kept/reused, because they're strictly stronger than invariant checks *and* cheap
to write correctly. They are not generalized to the more complex variants (adaptive/eigen-projection
/FFT/folding) — there, hand-deriving correct goldens is impractical and the result would be brittle;
invariant checks are the right tool.

## What drives the tests: the real production seam, end to end

`tests/support/pipeline_harness.hpp`'s `SyntheticPipelineRun<Config>` wires a real
`ProcessorState<Config>` to a real `Lambda*Pipeline<Config>` and a real `Output`, via the exact
seam production code uses:

```
PacketInput → ProcessorState → GPUPipeline → Output
```

Concretely: it feeds synthetic on-the-wire packets through `ProcessorState`'s normal ingestion
path (`get_current_write_pointer` / `add_received_packet_metadata` / `get_next_write_pointer`,
the same calls `KernelSocketPacketCapture`/`PCAPPacketCapture` make), sets
`synchronous_pipeline = true` so `ProcessorState::handle_buffer_completion()` calls
`pipeline_->execute_pipeline(...)` directly on the calling thread (no background threads/queues —
see `include/spatial/spatial.hpp`), and lets the result land in a real `SingleHostMemoryOutput<Config>`
for assertions to inspect.

This is intentionally *not* the same as `test_pipeline.cu`'s `DummyFinalPacketData`, which pokes
values directly into `LambdaConfig` array layouts, bypassing `ProcessorState` ingestion entirely —
that's a lower-level, more implementation-coupled shortcut. Driving the full real seam is what
"test at a high level" means here: it's the same path production traffic takes, so a test failure
means something that matters to a real run broke.

## The pieces (`tests/support/`)

- **`test_configs.hpp`** — canonical small `LambdaConfig` instantiations shared across tests,
  replacing the ad-hoc `Config`/`MultiFPGAConfig`/`TestConfig`/`MockT` aliases that different test
  files previously each defined for themselves:
  - `SmallSingleFPGAConfig` — 1 channel, 1 FPGA, 4 receivers: the minimal layout that exercises
    the full Tensor Core Correlator + ccglib GEMM path.
  - `SmallMultiFPGAConfig` — 1 channel, 3 FPGAs (2 receivers/packet each): exercises FPGA-to-FPGA
    delay alignment and multi-source reassembly.
  - `SmallTwoChannelConfig` — 2 channels, 1 FPGA: exercises multi-channel output independence (feed
    distinct data per channel, assert no cross-channel contamination).
  - `SmallTwoPacketConfig` — 1 channel, 2 packets for correlation: exercises accumulation over
    multiple packets (autocorrelation power doubles relative to single-packet).
- **`synthetic_packets.hpp`** — `build_lambda_wire_packet<Config>(...)` /
  `feed_lambda_packet<Config>(...)`: builds a correctly-laid-out Ethernet+IP+UDP+Custom+Payload
  wire packet (and, for the latter, pushes it through a real `ProcessorStateBase`'s write-pointer
  protocol), parameterized by `sample_fn(time, receiver, pol)` / `scale_fn(receiver, pol)`
  generator callbacks so callers can inject whatever data pattern they need (constant, tone, ramp,
  ...) without duplicating wire-format byte-layout logic. Consolidates what used to be two
  near-duplicate builders in `test_processor.cu` and `test_packet_formats.cpp`.
- **`pipeline_harness.hpp`** — `SyntheticPipelineRun<Config>` (described above), plus
  `make_unity_beam_weights<Config>()` / `make_unity_antenna_gains<Config>()` ("do nothing to the
  signal" baselines), plus `pipeline_factories::make_*_pipeline<Config>(...)` — small per-variant
  construction helpers. A single generic factory isn't realistic because the 7 pipeline classes
  take genuinely different constructor arguments (e.g. `LambdaGPUPipeline(int, BeamWeightsT<T>*)`
  vs `LambdaAntennaSpectraPipeline(int)` vs `LambdaProjectionPipeline<T, NR_EIG, NR_RUNS>(int)`).
  `make_gpu_pipeline` and `make_corr_beam_only_pipeline` are implemented; add the rest incrementally
  as tests for those variants are written.
- **`assertions.hpp`** — the property/invariant checks themselves:
  `assert_all_finite` (catches NaN/Inf from uninitialized memory, bad FFT plans, eigendecomposition
  blowups, ...), `assert_autocorrelation_invariants` (same-polarization autocorrelations are real
  and non-negative; each receiver's polarization covariance matrix is Hermitian — see the note
  below on why this isn't full baseline-level Hermitian symmetry), `assert_eigenvalues_ascending_nonnegative`
  (cuSOLVER's `cusolverDnXsyevBatched` contract), `assert_tone_detected` (feed a known-frequency
  tone, assert the FFT peak lands at the expected bin — robust to scaling/normalization changes).
- **`test_harness_selftest.cu`** — `PipelineHarnessSelfTest`, proving the harness is wired
  correctly by reproducing `test_pipeline.cu::Ex1`'s hand-derived exact values
  (`beam_data == (8, -8)`, `visibilities == (64, 0)`) end to end through the harness, *and*
  demonstrating the invariant-based style on the same run.

## A gotcha this surfaced: packed triangular baseline storage

The plan originally called for a generic `assert_hermitian_symmetric(visibilities)` checking
`V[baseline(i,j)] ≈ conj(V[baseline(j,i)])`. That's not directly testable: the Tensor Core
Correlator stores visibilities in **packed lower-triangular** form —
`baseline_index(i, j) = j*(j+1)/2 + i`, valid only for `i <= j` (see `storeVisibility` in
`extern/tcc/libtcc/kernel/TCCorrelator.cu`) — so there is no stored slot for the conjugate pair
`(j, i)` when `j > i` to compare against. `assert_autocorrelation_invariants` checks the subset of
"Hermitian visibilities" that *does* survive this storage scheme: every receiver's own
per-polarization covariance block `V[baseline(r,r)][p][q]` is square and must still be Hermitian
PSD, regardless of how the correlator lays out cross-baseline pairs.

## A gotcha this surfaced: a likely latent race on `Output::arrivals`

An earlier version of the self-test asserted that `Output::arrivals` reflected a fully-arrived
synthetic buffer (`assert_all_packets_arrived`, since removed). It failed deterministically: the
array came back all-`false`. Tracing it — `LambdaPipelineIngest::ingest_and_scale`
(`include/spatial/pipeline.hpp`) queues `release_buffer_host_func` via `cudaLaunchHostFunc` on a
per-buffer host stream *immediately* on entry; that callback calls `ProcessorState::release_buffer`,
which `memset`s the very same buffer's `arrivals` array to zero in preparation for reuse — but the
`std::memcpy` that copies `arrivals` out to `Output` happens ~150 lines later in the same
`execute_pipeline`, with no synchronization between the two. With small/fast configs (i.e. exactly
the kind tests use), the async callback reliably wins that race and zeroes the array first.

This looks like a genuine pre-existing bug independent of the harness — `Output::arrivals` may be
unreliable in production too, just masked there by larger configs giving the host thread more of a
head start. Fixing it is real production-code work and out of scope for the test harness itself;
it's flagged here so whoever picks it up has the trace already done. (If/when it's fixed,
`assert_all_packets_arrived`-style precondition checks — "did the synthetic feed actually produce
a complete buffer before asserting on pipeline output" — would be worth re-adding.)

## A gotcha this surfaced: uninitialized pinned memory across test lifecycles

`LambdaFinalPacketData`'s constructor allocates pinned host memory via `cudaHostAlloc`, which does
**not** zero-initialize. In production this is harmless — the object is created once at startup and
the pipeline fills every slot before it's read. In tests, where many `ProcessorState` objects are
created and destroyed in sequence, the CUDA pinned-memory pool recycles addresses; a subsequent
allocation at the same address inherits stale bytes from the prior test.

The symptom: `MultipleFPGAPlacementWithDifferentIDTest` failed non-deterministically when two
other processor tests ran before it, leaving non-zero values in `samples` slots that should have
been zero (delay-alignment padding). The fix — `memset(samples/scales/arrivals, 0, sizeof(...))` in
the constructor (`include/spatial/packet_formats.hpp`) — makes the constructor's post-condition
match production assumptions and eliminates the cross-test dependency.

## Adding a test for a new pipeline variant

1. If needed, add a `pipeline_factories::make_<variant>_pipeline<Config>(...)` to
   `pipeline_harness.hpp` (the variant's constructor signature dictates the shape).
2. Pick (or add) a `Config` from `test_configs.hpp` sized for the variant under test.
3. `SyntheticPipelineRun<Config> driver(*pipeline, output, ...);` then
   `driver.run(sample_fn, scale_fn)` (or `run_uniform` if the data pattern doesn't depend on
   channel/FPGA/packet index).
4. Assert on `*output-><field>` using the helpers in `assertions.hpp` — reach for invariant checks
   first; only write exact-value goldens if the variant's math is genuinely hand-derivable (as
   `LambdaGPUPipeline`'s is).
