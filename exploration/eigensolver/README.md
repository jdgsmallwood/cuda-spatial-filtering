# Eigendecomposition speedup exploration

Status: **exploratory, not integrated into the pipeline.** Kept buildable (wired into
`apps/CMakeLists.txt`) so it doesn't bit-rot, but nothing here is called from
`LambdaGPUPipeline` or any production code path. Resume by reading this file first.

## Why this exists

Benchmarking `LambdaGPUPipeline` with fine channelization on (`NR_FINE_CHANNELS=32`)
showed a ~2x per-buffer slowdown vs. off, and stage-level timing (see
`/home/ubuntu/.claude/plans/i-want-to-start-breezy-lampson.md`'s benchmarking history,
`apps/bench_gpu.cu --fine-channel-check`) isolated **eigendecomposition
(`cusolverDnXsyevBatched`) as 80-88% of that added time** -- correlation and
beamforming barely grow. This directory is the investigation into whether that
specific stage can be sped up. Two real findings from earlier in the same session,
folded in here for continuity:

- The dead FFT/fine-delay buffers `LambdaGPUPipeline` used to allocate unconditionally
  even when `NR_FINE_CHANNELS > 1` made them unreachable dead code -- fixed and
  **committed already** (`pipeline/common.hpp`'s `make_device_ptr_if`,
  `pipeline/lambda_gpu_pipeline.hpp`). Not part of this exploration; mentioned because
  it's what got `num_buffers=2` fitting in VRAM at all for the configs benchmarked here.
- `LambdaGPUPipeline`'s eigendecomposition result (`eigenvalues`/`decomp_visibilities`)
  is **never fed back into the beamforming weights** -- traced `update_weights()`
  (`src/spatial.cu`) and confirmed the kernel it launches ignores the eigenvalue/
  eigenvector parameters entirely; the only consumer is the HDF5 output writer
  (offline diagnostic/RFI-monitoring product). This matters for scope: nothing about
  correctness of the *correlator/beamformer output* depends on this decomposition being
  fast, exact, or even running every buffer.

## Files

- `bench_eigensolver.cu` -- standalone benchmark, no pipeline/TCC/ccglib dependency
  (just cuSOLVER + this directory's custom kernel), comparing four approaches at the
  real problem's matrix size (`n=NR_RECEIVERS`, default 40) and batch sizes
  (`CUSOLVER_BATCH_SIZE = NR_CHANNELS * NR_POLARIZATIONS^2`, using the same
  `Cfg{8,16}ch4fpga{,_fine32}` configs as `apps/bench_gpu.cu`'s `--fine-channel-check`).
  Build target: `bench_eigensolver` (already wired into `apps/CMakeLists.txt`).
- `batched_topk_eigensolver.cuh` -- the custom kernel (see below).

## What was tried

### 1. `cusolverDnCheevjBatched` (cuSOLVER's batched Jacobi eigensolver) -- ruled out

Purpose-built for "many small identical Hermitian matrices," which looked like a good
API match. Tested at default tolerance, 1e-3, 1e-2 (capped 15 sweeps), and 1e-1 (capped
3 sweeps). **Slower than the current `cusolverDnXsyevBatched` at every setting that
preserves real accuracy** (0.73-0.98x, i.e. up to 30% slower); only "wins" (1.3-1.4x) at
tolerance=1e-1/3 sweeps, where eigenvalue error is ~0.15 absolute -- too coarse to trust.
Reproduce: `./bench_eigensolver --tolerance <tol> --max-sweeps <n>` (uses
`make_hermitian_batch`'s generic, non-PSD test matrices -- fine for this comparison
since it's solver-vs-solver on identical input, not a top-K accuracy question).

### 2. `cusolverDnXsyevdx` (top-K via index-range selection) -- ruled out, crashes

Not a batched API (one host-driven call per matrix -- already a concern: batch=896-1792
means hundreds of host-driven calls per buffer). Segfaults **inside cuSOLVER itself**
(confirmed via `gdb` backtrace, not application code) for this complex-Hermitian +
`CUSOLVER_EIG_RANGE_I` combination, in this container's cuSOLVER build. Tried the two
most likely fixes (device- vs host-memory for the `meig` output parameter; non-null
`vl`/`vu` despite being documented as unused in index-range mode) -- both real,
principled attempts, still crashes. Left disabled by default
(`--run-syevdx` to re-enable and hit the crash again). Given it's not batched anyway,
this was already a weak candidate even before the crash.

### 3. Custom batched top-K solver (`batched_topk_eigensolver.cuh`) -- the promising one

**Method**: blocked subspace (orthogonal) iteration -- repeatedly multiply an N x K seed
by A and re-orthonormalize (modified Gram-Schmidt) -- converges Q's column space to A's
K-dominant-eigenvalue invariant subspace, then a Rayleigh-Ritz refinement (eigendecompose
the tiny K x K matrix `Q^H A Q` via a from-scratch cyclic Jacobi solver, rotate Q by the
result) recovers proper eigenpairs, not just the subspace. One CUDA thread block per
matrix, everything in shared memory; the parallel-across-the-batch dimension is where
throughput comes from, not intra-block parallelism (Gram-Schmidt and the K x K Jacobi
solve are currently single-threaded within a block -- see "Known limitations" below).

**Three real bugs found and fixed** during validation (all three were plausible-looking
code that produced *wrong answers*, not crashes -- caught by adding a residual check
`||A*v - lambda*v|| / ||v||`, not just comparing eigenvalues against a reference, since
eigenvalues alone can look right while eigenvectors are wrong):

1. Off-diagonal Hermitian-update term in the cyclic Jacobi solver had an erroneous
   `conj()` -- traced by hand-deriving `H' = G^H H G` from scratch.
2. Sign error in the Jacobi rotation-angle formula (`t = sign(zeta)/(...)` should be
   `t = -sign(zeta)/(...)`) -- caught by an isolated host-side C++ test against a
   hand-computable 2x2 case (eigenvalues of `[[1,2+i],[2-i,3]]` are known in closed
   form), where the rotation provably didn't zero the off-diagonal entry it was
   supposed to.
3. Plain subspace iteration on `A` converges to the eigenvectors of largest
   **magnitude**, not largest **value** -- wrong for matrices with negative eigenvalues.
   Fixed with a Gershgorin-bound shift applied only during iteration (the final
   Rayleigh-Ritz step uses the true unshifted `A`, so reported eigenvalues are exact).
   Confirmed via user input that real visibility matrices are PSD (all eigenvalues >=0)
   so this specific failure mode wouldn't have bitten the real use case -- kept the fix
   anyway since it's cheap and makes the kernel correct regardless of input.

After all three fixes: residual ~1e-5 to 2e-5 (machine precision for float32) on both a
hand-computable case and the real problem size, confirming genuine correctness, not
"eigenvalues happen to look close."

## The key finding: it depends entirely on K matching the *real* number of separated modes

Initial testing used `k=10` (the user's original "first ~10 of 40" ask) against a
synthetic PSD test matrix (`make_psd_visibility_batch`: `num_sources` random outer
products + a flat `noise_floor*I`, modeling "a few strong RFI/signal directions over a
noise floor" -- confirmed representative of real visibility matrix structure). With
`num_sources=3`, asking for the top 10 forces the algorithm to also resolve 7 more
directions from **37 numerically-tied eigenvalues** (the flat noise floor) -- there is no
real signal to converge to there, so it doesn't, regardless of iteration count (tried up
to 300 iterations, still not converged) -- and even where it *did* eventually converge
(k=10 was tested at earlier, smaller n first), the custom solver was **6-11x slower**
than `cusolverDnXsyevBatched`, both because of the many iterations needed and because
Gram-Schmidt/the Jacobi solve are single-threaded (see limitations below).

Setting `k=3` (matching `num_sources`, i.e. asking only for the genuinely separated
modes) changes the picture completely -- convergence is fast and there's a **real,
validated speedup, growing with batch size**:

| Iterations | Eigenvector residual | Speedup @ batch=1792 (16ch, fine channelization on) |
|---|---|---|
| 35 | 2.6% | **1.94x** |
| 45 | 0.18% | **1.65x** |
| 55 | ~0.01% (machine precision) | **1.41x** |

Speedup is smallest (~1.1x) at the smallest batch (32) and largest at the biggest batch
(1792) -- which is exactly the fine-channelized "on" case that was the original
motivation, a fortunate alignment. Reproduce:
```
./bench_eigensolver --skip-jacobi --n 40 --k 3 --subspace-iterations 45 \
    --num-sources 3 --noise-floor 0.1 --timed-runs 5 --warmup 1
```

## Open question (blocks any further investment)

**What is the real number of well-separated dominant eigenmodes in production
visibility data** -- closer to the `num_sources=3` this benchmark validated a win at, or
closer to the `k=10` that showed a 6-11x loss? This is a domain/data question, not
something resolvable from this synthetic benchmark. Until it's answered:

- If real data has ~3-5 genuinely separated modes (with the rest sitting in a
  noise-floor-like cluster): this approach is worth pursuing further.
- If real data has closer to 10 genuinely separated modes, or no flat noise-floor
  structure at all: this approach likely isn't worth it as currently implemented.

## Known limitations / next steps if resumed

- **Gram-Schmidt and the K x K Jacobi solve are single-threaded** within each block (a
  deliberate simplification to reduce bug risk while getting a first *correct* version
  -- see how much debugging the three bugs above took even with this simplification).
  Parallelizing Gram-Schmidt (block-cooperative or warp-shuffle reduction instead of one
  thread doing all the O(n*k) work serially) is the most promising next lever if the
  open question above resolves favorably -- most of the block's 256 threads currently
  sit idle during that phase, which runs every iteration.
- Only tested against synthetic data (`make_psd_visibility_batch`); needs validation
  against real captured visibility matrices before trusting the convergence-speed
  numbers, since real RFI environments may have a messier eigenvalue structure than the
  clean "N sources + flat noise floor" model used here.
- Not tested at `num_buffers > 1` or inside the actual pipeline -- this is a standalone
  cuSOLVER-vs-custom-kernel microbenchmark, not an integration.
- If pursued to integration: would need to replace the
  `cusolverDnXsyevBatched` call in `include/spatial/pipeline/lambda_gpu_pipeline.hpp`
  (`execute_pipeline`, the `pre_corr_done`/`eigen_done` stage-boundary events already
  added there make it easy to re-measure the isolated eigen-stage cost in place), and
  the output writer path (`register_eigendecomposition_data_block`,
  `get_eigenvectors_data_landing_pointer`) would need updating for K-sized output
  instead of N-sized.
