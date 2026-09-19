#pragma once

#include <cuComplex.h>

// Batched top-K eigensolver for many small, identical-size complex Hermitian
// matrices -- the shape LambdaGPUPipeline's cuSOLVER call operates in
// (N=NR_RECEIVERS~40, batch=NR_CHANNELS*NR_POLARIZATIONS^2 up to ~1800). See
// the investigation in /home/ubuntu/.claude/plans/i-want-to-start-breezy-lampson.md's
// eigendecomposition follow-up: neither cusolverDnXsyevBatched (full N)
// nor cusolverDnCheevjBatched (batched Jacobi, also full N) nor
// cusolverDnXsyevdx (unbatched top-K, and crashes for this input shape in
// this cuSOLVER build) gave a validated win, so this is a from-scratch
// custom kernel exploiting "only need the top K << N eigenpairs."
//
// Method: blocked subspace (orthogonal) iteration -- repeatedly multiply an
// N x K seed by A and re-orthonormalize -- converges the column space of Q
// to the invariant subspace of A's K dominant eigenvalues, at a rate set by
// the eigenvalue gap between the K-th and (K+1)-th eigenvalues. A final
// Rayleigh-Ritz step (eigendecompose the tiny K x K matrix Q^H*A*Q via
// cyclic Jacobi, then rotate Q by the Ritz vectors) recovers proper
// eigenvalues/eigenvectors, not just the subspace.
//
// One thread block per matrix. All per-matrix state lives in shared memory
// for the whole iteration (single global read of A, single global write of
// the K results) -- everything here is deliberately tiny (N~40, K~10), so
// the sequential parts (Gram-Schmidt, the K x K Jacobi solve) run on a
// single thread per block; only the genuinely parallel N*K matrix-vector
// product A*Q is spread across the block's threads.
//
// NOT numerically bulletproof: modified Gram-Schmidt (not Householder QR)
// loses orthogonality faster under ill-conditioning, and subspace iteration
// converges slowly when the K/K+1 eigenvalue gap is small. Validate against
// a trusted full decomposition (cusolverDnXsyevBatched) before trusting
// this for anything beyond a benchmark -- see bench_eigensolver.cu's
// accuracy checks (eigenvalues within tolerance AND ||A*v - lambda*v||
// residual AND eigenvector orthonormality).

namespace batched_topk {

__device__ __forceinline__ void gram_schmidt_orthonormalize(cuComplex *Qmat, int n, int k) {
  for (int c = 0; c < k; ++c) {
    for (int p = 0; p < c; ++p) {
      cuComplex dot = make_cuComplex(0.f, 0.f);
      for (int i = 0; i < n; ++i) {
        dot = cuCaddf(dot, cuCmulf(cuConjf(Qmat[i + p * n]), Qmat[i + c * n]));
      }
      for (int i = 0; i < n; ++i) {
        Qmat[i + c * n] = cuCsubf(Qmat[i + c * n], cuCmulf(dot, Qmat[i + p * n]));
      }
    }
    float norm2 = 0.f;
    for (int i = 0; i < n; ++i) {
      const cuComplex v = Qmat[i + c * n];
      norm2 += v.x * v.x + v.y * v.y;
    }
    const float inv_norm = 1.0f / fmaxf(sqrtf(norm2), 1e-20f);
    for (int i = 0; i < n; ++i) {
      Qmat[i + c * n] = make_cuComplex(Qmat[i + c * n].x * inv_norm, Qmat[i + c * n].y * inv_norm);
    }
  }
}

// In-place cyclic Jacobi eigensolve of a small k x k complex Hermitian
// matrix H (column-major, overwritten). eigvals gets ascending eigenvalues;
// eigvecs (must be pre-sized k*k) gets the corresponding eigenvector columns.
__device__ __forceinline__ void small_hermitian_jacobi(cuComplex *H, int k, float *eigvals,
                                                        cuComplex *eigvecs) {
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      eigvecs[i + j * k] = make_cuComplex((i == j) ? 1.f : 0.f, 0.f);
    }
  }

  constexpr int MAX_SWEEPS = 30;
  constexpr float EPS = 1e-7f;
  for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
    float off_norm2 = 0.f;
    for (int p = 0; p < k - 1; ++p) {
      for (int q = p + 1; q < k; ++q) {
        const cuComplex w = H[p + q * k];
        const float wmag = cuCabsf(w);
        off_norm2 += wmag * wmag;
        if (wmag < 1e-20f) continue;

        const float a = H[p + p * k].x;
        const float b = H[q + q * k].x;
        const float theta = -atan2f(w.y, w.x); // phase correction so the (p,q) entry becomes real
        const float zeta = (b - a) / (2.0f * wmag);
        float t;
        if (fabsf(zeta) < 1e-12f) {
          t = -1.0f;
        } else {
          // Solving (d-a)cs + b(c^2-s^2) = 0 for t=s/c gives b*t^2 - (d-a)*t - b = 0
          // (verified by hand and against a 2x2 hand-computable case -- an
          // earlier version of this had the (d-a)t term's sign backwards,
          // which produces a `t` that does NOT actually zero the (p,q)
          // entry despite solving a plausible-looking quadratic). The
          // stable small root is t = -sign(zeta)/(|zeta|+sqrt(zeta^2+1)),
          // not +sign(zeta)/(...).
          const float sign_zeta = (zeta >= 0.f) ? 1.0f : -1.0f;
          t = -sign_zeta / (fabsf(zeta) + sqrtf(zeta * zeta + 1.0f));
        }
        const float c = 1.0f / sqrtf(t * t + 1.0f);
        const float s = t * c;
        const cuComplex e_itheta = make_cuComplex(cosf(theta), sinf(theta));
        // Unitary rotation G = diag(1, e^{i theta}) * [[c,-s],[s,c]], restricted
        // to rows/cols (p,q); zeros H[p][q] under the similarity G^H H G.
        const cuComplex G00 = make_cuComplex(c, 0.f);
        const cuComplex G01 = make_cuComplex(-s, 0.f);
        const cuComplex G10 = cuCmulf(make_cuComplex(s, 0.f), e_itheta);
        const cuComplex G11 = cuCmulf(make_cuComplex(c, 0.f), e_itheta);

        for (int i = 0; i < k; ++i) {
          if (i == p || i == q) continue;
          // H' = G^H * H * G; for row i outside {p,q}, G^H's row i collapses
          // to the identity, so H'[i][p]/H'[i][q] are just H[i][p..q] * G's
          // column p/q directly -- no conjugate here (that's only for the
          // (p,q) 2x2 block update below, where G^H genuinely acts on both
          // sides). Verified by hand against H'=G^H H G and cross-checked
          // via the ||A*v - lambda*v|| residual check in bench_eigensolver.cu.
          const cuComplex Hip = H[i + p * k];
          const cuComplex Hiq = H[i + q * k];
          const cuComplex new_ip = cuCaddf(cuCmulf(Hip, G00), cuCmulf(Hiq, G10));
          const cuComplex new_iq = cuCaddf(cuCmulf(Hip, G01), cuCmulf(Hiq, G11));
          H[i + p * k] = new_ip;
          H[p + i * k] = cuConjf(new_ip);
          H[i + q * k] = new_iq;
          H[q + i * k] = cuConjf(new_iq);
        }

        const cuComplex Hpp = H[p + p * k], Hpq = H[p + q * k];
        const cuComplex Hqp = H[q + p * k], Hqq = H[q + q * k];
        const cuComplex T00 = cuCaddf(cuCmulf(Hpp, G00), cuCmulf(Hpq, G10));
        const cuComplex T01 = cuCaddf(cuCmulf(Hpp, G01), cuCmulf(Hpq, G11));
        const cuComplex T10 = cuCaddf(cuCmulf(Hqp, G00), cuCmulf(Hqq, G10));
        const cuComplex T11 = cuCaddf(cuCmulf(Hqp, G01), cuCmulf(Hqq, G11));
        const cuComplex M00 = cuCaddf(cuCmulf(cuConjf(G00), T00), cuCmulf(cuConjf(G10), T10));
        const cuComplex M01 = cuCaddf(cuCmulf(cuConjf(G00), T01), cuCmulf(cuConjf(G10), T11));
        const cuComplex M10 = cuCaddf(cuCmulf(cuConjf(G01), T00), cuCmulf(cuConjf(G11), T10));
        const cuComplex M11 = cuCaddf(cuCmulf(cuConjf(G01), T01), cuCmulf(cuConjf(G11), T11));
        H[p + p * k] = make_cuComplex(M00.x, 0.f);
        H[p + q * k] = M01;
        H[q + p * k] = M10;
        H[q + q * k] = make_cuComplex(M11.x, 0.f);

        for (int i = 0; i < k; ++i) {
          const cuComplex Vip = eigvecs[i + p * k];
          const cuComplex Viq = eigvecs[i + q * k];
          eigvecs[i + p * k] = cuCaddf(cuCmulf(Vip, G00), cuCmulf(Viq, G10));
          eigvecs[i + q * k] = cuCaddf(cuCmulf(Vip, G01), cuCmulf(Viq, G11));
        }
      }
    }
    if (off_norm2 < EPS * EPS) break;
  }

  for (int i = 0; i < k; ++i) eigvals[i] = H[i + i * k].x;
  // Selection sort ascending (k is tiny -- O(k^2) is negligible), swapping
  // eigenvector columns in lockstep.
  for (int i = 0; i < k - 1; ++i) {
    int min_idx = i;
    for (int j = i + 1; j < k; ++j) {
      if (eigvals[j] < eigvals[min_idx]) min_idx = j;
    }
    if (min_idx != i) {
      const float tmp = eigvals[i];
      eigvals[i] = eigvals[min_idx];
      eigvals[min_idx] = tmp;
      for (int r = 0; r < k; ++r) {
        const cuComplex t = eigvecs[r + i * k];
        eigvecs[r + i * k] = eigvecs[r + min_idx * k];
        eigvecs[r + min_idx * k] = t;
      }
    }
  }
}

// One block per matrix. Shared memory layout (all cuComplex except where
// noted): A[n*n], Q[n*k], AQ[n*k], H[k*k], ritz_vecs[k*k], ritz_vals[k]
// (float). Caller must launch with shared memory bytes =
// shared_mem_bytes(n, k) and blockDim.x >= 32 (more threads only help the
// A*Q parallel step; anything above n*k=400-ish stops helping).
inline size_t shared_mem_bytes(int n, int k) {
  return static_cast<size_t>(n) * n * sizeof(cuComplex) +          // A
        2ull * n * k * sizeof(cuComplex) +                        // Q, AQ
        2ull * k * k * sizeof(cuComplex) +                        // H, ritz_vecs
        static_cast<size_t>(k) * sizeof(float);                   // ritz_vals
}

__global__ void batched_topk_kernel(const cuComplex *__restrict__ A_batch,
                                    float *__restrict__ eigvals_out,
                                    cuComplex *__restrict__ eigvecs_out, int n, int k,
                                    int iterations, unsigned seed_base) {
  extern __shared__ unsigned char smem_raw[];
  cuComplex *A = reinterpret_cast<cuComplex *>(smem_raw);
  cuComplex *Q = A + static_cast<size_t>(n) * n;
  cuComplex *AQ = Q + static_cast<size_t>(n) * k;
  cuComplex *H = AQ + static_cast<size_t>(n) * k;
  cuComplex *ritz_vecs = H + static_cast<size_t>(k) * k;
  float *ritz_vals = reinterpret_cast<float *>(ritz_vecs + static_cast<size_t>(k) * k);

  const int b = blockIdx.x;
  const cuComplex *A_global = A_batch + static_cast<size_t>(b) * n * n;

  for (int idx = threadIdx.x; idx < n * n; idx += blockDim.x) A[idx] = A_global[idx];
  __syncthreads();

  if (threadIdx.x == 0) {
    unsigned int rng = seed_base + static_cast<unsigned int>(b) * 2654435761u + 1u;
    for (int idx = 0; idx < n * k; ++idx) {
      rng = rng * 1664525u + 1013904223u;
      const float re = static_cast<float>((rng >> 8) & 0xFFFFu) / 65536.0f - 0.5f;
      rng = rng * 1664525u + 1013904223u;
      const float im = static_cast<float>((rng >> 8) & 0xFFFFu) / 65536.0f - 0.5f;
      Q[idx] = make_cuComplex(re, im);
    }
    gram_schmidt_orthonormalize(Q, n, k);
  }
  __syncthreads();

  // Gershgorin-bound shift: plain subspace iteration on A converges to the
  // eigenvectors of largest |lambda|, not largest lambda -- for a matrix
  // with negative eigenvalues (possible in general, though the real
  // pipeline's visibility/covariance matrices are positive semi-definite so
  // this wouldn't actually bite there) those can disagree. Iterating on
  // A+shift*I instead (same eigenvectors, eigenvalues shifted by a constant)
  // with shift >= the spectral radius guarantees every shifted eigenvalue is
  // positive, so "largest magnitude" and "largest value" coincide. Applied
  // only during iteration -- the final Rayleigh-Ritz step below uses the
  // original unshifted A, so reported eigenvalues are exact, not shifted.
  __shared__ float shift;
  if (threadIdx.x == 0) {
    float max_row_sum = 0.f;
    for (int i = 0; i < n; ++i) {
      float row_sum = 0.f;
      for (int j = 0; j < n; ++j) row_sum += cuCabsf(A[i + j * n]);
      max_row_sum = fmaxf(max_row_sum, row_sum);
    }
    shift = max_row_sum;
  }
  __syncthreads();

  for (int iter = 0; iter < iterations; ++iter) {
    for (int idx = threadIdx.x; idx < n * k; idx += blockDim.x) {
      const int i = idx % n, c = idx / n;
      cuComplex sum = make_cuComplex(0.f, 0.f);
      for (int j = 0; j < n; ++j) {
        sum = cuCaddf(sum, cuCmulf(A[i + j * n], Q[j + c * n]));
      }
      sum = cuCaddf(sum, make_cuComplex(shift * Q[idx].x, shift * Q[idx].y)); // + shift*I*Q
      AQ[idx] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      gram_schmidt_orthonormalize(AQ, n, k);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < n * k; idx += blockDim.x) Q[idx] = AQ[idx];
    __syncthreads();
  }

  // Rayleigh-Ritz: AQ = A*Q (parallel), H = Q^H * AQ (serial, tiny), then
  // eigendecompose H and rotate Q by its eigenvectors.
  for (int idx = threadIdx.x; idx < n * k; idx += blockDim.x) {
    const int i = idx % n, c = idx / n;
    cuComplex sum = make_cuComplex(0.f, 0.f);
    for (int j = 0; j < n; ++j) sum = cuCaddf(sum, cuCmulf(A[i + j * n], Q[j + c * n]));
    AQ[idx] = sum;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int a = 0; a < k; ++a) {
      for (int c = 0; c < k; ++c) {
        cuComplex sum = make_cuComplex(0.f, 0.f);
        for (int i = 0; i < n; ++i) {
          sum = cuCaddf(sum, cuCmulf(cuConjf(Q[i + a * n]), AQ[i + c * n]));
        }
        H[a + c * k] = sum;
      }
    }
    small_hermitian_jacobi(H, k, ritz_vals, ritz_vecs);
  }
  __syncthreads();

  cuComplex *V_out = eigvecs_out + static_cast<size_t>(b) * n * k;
  for (int idx = threadIdx.x; idx < n * k; idx += blockDim.x) {
    const int i = idx % n, c = idx / n;
    cuComplex sum = make_cuComplex(0.f, 0.f);
    for (int a = 0; a < k; ++a) sum = cuCaddf(sum, cuCmulf(Q[i + a * n], ritz_vecs[a + c * k]));
    V_out[idx] = sum;
  }
  if (threadIdx.x < k) {
    eigvals_out[static_cast<size_t>(b) * k + threadIdx.x] = ritz_vals[threadIdx.x];
  }
}

} // namespace batched_topk
