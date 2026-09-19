// Standalone microbenchmark: cusolverDnXsyevBatched (current, LambdaGPUPipeline's
// generic 64-bit batched solver) vs. cusolverDnCheevjBatched (candidate, cuSOLVER's
// purpose-built batched Jacobi eigensolver for many small identical Hermitian
// matrices -- exactly this problem's shape). See the investigation in
// /home/ubuntu/.claude/plans/i-want-to-start-breezy-lampson.md's eigendecomposition
// follow-up: LambdaGPUPipeline's eigendecomposition dominates the fine-channelization
// slowdown (~80-88% of the added per-buffer time), so this checks whether a different
// cuSOLVER API is meaningfully faster at the same matrix size (N=NR_RECEIVERS) and
// batch sizes (NR_CHANNELS * NR_POLARIZATIONS^2) this pipeline actually uses.
//
// No pipeline/TCC/ccglib dependency -- pure cuSOLVER, so this builds and runs fast
// and in isolation from everything else.
#include "batched_topk_eigensolver.cuh"
#include "spatial/logging.hpp"
#include <argparse/argparse.hpp>
#include <chrono>
#include <cmath>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <random>
#include <vector>

namespace {

// Fills a batch of random NxN Hermitian matrices, column-major, upper
// triangle meaningful (both solvers below are called with uplo=UPPER).
// Generic (mixed-sign eigenvalues, flat spectrum) -- NOT representative of
// the real pipeline's visibility matrices (see make_psd_visibility_batch
// below), kept for the accuracy cross-check against cuSOLVER's full
// decomposition (which doesn't care about eigenvalue structure).
std::vector<cuComplex> make_hermitian_batch(int n, int batch_size, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<cuComplex> host(static_cast<size_t>(n) * n * batch_size);
  for (int b = 0; b < batch_size; ++b) {
    cuComplex *A = host.data() + static_cast<size_t>(b) * n * n;
    for (int i = 0; i < n; ++i) {
      A[i + i * n] = make_cuComplex(dist(rng), 0.0f); // real diagonal
      for (int j = i + 1; j < n; ++j) {
        const float re = dist(rng), im = dist(rng);
        A[i + j * n] = make_cuComplex(re, im);  // upper triangle (row i, col j)
        A[j + i * n] = make_cuComplex(re, -im); // lower = conjugate (for completeness)
      }
    }
  }
  return host;
}

// Positive semi-definite, representative of real visibility/covariance
// matrices: A = sum_{s=1}^{S} outer(v_s, v_s) + noise_floor*I, for S random
// complex "source" vectors -- S dominant eigenvalues (roughly ||v_s||^2,
// well-separated if S << n) sitting above a flat noise floor of the
// remaining n-S eigenvalues, exactly the "few strong sources + noise"
// structure a real RFI/signal covariance matrix has. This is the favorable
// case for subspace iteration (the S/S+1 eigenvalue gap, not the n/n+1
// one, sets the convergence rate) -- confirmed per-user real data has no
// negative eigenvalues, unlike make_hermitian_batch's generic case above.
std::vector<cuComplex> make_psd_visibility_batch(int n, int batch_size, int num_sources,
                                                 float noise_floor, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<cuComplex> host(static_cast<size_t>(n) * n * batch_size, make_cuComplex(0.f, 0.f));
  std::vector<cuComplex> v(n);
  for (int b = 0; b < batch_size; ++b) {
    cuComplex *A = host.data() + static_cast<size_t>(b) * n * n;
    for (int i = 0; i < n; ++i) A[i + i * n] = make_cuComplex(noise_floor, 0.f);
    for (int s = 0; s < num_sources; ++s) {
      for (int i = 0; i < n; ++i) v[i] = make_cuComplex(dist(rng), dist(rng));
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          // A += v * v^H
          A[i + j * n] = cuCaddf(A[i + j * n], cuCmulf(v[i], cuConjf(v[j])));
        }
      }
    }
    // Force exact Hermitian symmetry (accumulated rounding can leave tiny
    // asymmetry) and real diagonal.
    for (int i = 0; i < n; ++i) {
      A[i + i * n] = make_cuComplex(A[i + i * n].x, 0.f);
      for (int j = i + 1; j < n; ++j) {
        const cuComplex avg = make_cuComplex(0.5f * (A[i + j * n].x + A[j + i * n].x),
                                             0.5f * (A[i + j * n].y - A[j + i * n].y));
        A[i + j * n] = avg;
        A[j + i * n] = cuConjf(avg);
      }
    }
  }
  return host;
}

struct SolverResult {
  double avg_ms = 0.0;
  std::vector<float> eigenvalues_first_matrix; // for accuracy cross-check
};

// Mirrors LambdaGPUPipeline's exact call pattern (lambda_gpu_pipeline.hpp).
SolverResult bench_xsyev(int n, int batch_size, const std::vector<cuComplex> &input,
                         int warmup, int timed_runs) {
  cusolverDnHandle_t handle;
  cusolverDnParams_t params;
  CUSOLVER_CHECK(cusolverDnCreate(&handle));
  CUSOLVER_CHECK(cusolverDnCreateParams(&params));

  cuComplex *d_A;
  float *d_W;
  int *d_info;
  const size_t matrix_elems = static_cast<size_t>(n) * n * batch_size;
  CUDA_CHECK(cudaMalloc(&d_A, matrix_elems * sizeof(cuComplex)));
  CUDA_CHECK(cudaMalloc(&d_W, static_cast<size_t>(n) * batch_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_info, batch_size * sizeof(int)));

  size_t work_device_size = 0, work_host_size = 0;
  CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
      handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n,
      CUDA_C_32F, d_A, n, CUDA_R_32F, d_W, CUDA_C_32F, &work_device_size,
      &work_host_size, batch_size));
  void *d_work = nullptr;
  CUDA_CHECK(cudaMalloc(&d_work, work_device_size));
  std::vector<uint8_t> h_work(work_host_size);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto run_once = [&]() {
    CUDA_CHECK(cudaMemcpy(d_A, input.data(), matrix_elems * sizeof(cuComplex),
                          cudaMemcpyHostToDevice));
    CUSOLVER_CHECK(cusolverDnXsyevBatched(
        handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n,
        CUDA_C_32F, d_A, n, CUDA_R_32F, d_W, CUDA_C_32F, d_work, work_device_size,
        h_work.data(), work_host_size, d_info, batch_size));
  };

  for (int i = 0; i < warmup; ++i) run_once();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < timed_runs; ++i) run_once();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  SolverResult r;
  r.avg_ms = ms / timed_runs;
  r.eigenvalues_first_matrix.resize(n);
  CUDA_CHECK(cudaMemcpy(r.eigenvalues_first_matrix.data(), d_W, n * sizeof(float),
                       cudaMemcpyDeviceToHost));

  cudaFree(d_A);
  cudaFree(d_W);
  cudaFree(d_info);
  cudaFree(d_work);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  return r;
}

// Batched Jacobi eigensolver -- purpose-built for many small identical
// Hermitian matrices, unlike Xsyev's general-purpose 64-bit API.
SolverResult bench_cheevj(int n, int batch_size, const std::vector<cuComplex> &input,
                          int warmup, int timed_runs, double tolerance, int max_sweeps) {
  cusolverDnHandle_t handle;
  syevjInfo_t params;
  CUSOLVER_CHECK(cusolverDnCreate(&handle));
  CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&params));
  if (tolerance > 0.0) {
    CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(params, tolerance));
  }
  if (max_sweeps > 0) {
    CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(params, max_sweeps));
  }

  cuComplex *d_A;
  float *d_W;
  int *d_info;
  const size_t matrix_elems = static_cast<size_t>(n) * n * batch_size;
  CUDA_CHECK(cudaMalloc(&d_A, matrix_elems * sizeof(cuComplex)));
  CUDA_CHECK(cudaMalloc(&d_W, static_cast<size_t>(n) * batch_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_info, batch_size * sizeof(int)));

  int lwork = 0;
  CUSOLVER_CHECK(cusolverDnCheevjBatched_bufferSize(
      handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, d_A, n, d_W,
      &lwork, params, batch_size));
  cuComplex *d_work = nullptr;
  CUDA_CHECK(cudaMalloc(&d_work, static_cast<size_t>(lwork) * sizeof(cuComplex)));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto run_once = [&]() {
    CUDA_CHECK(cudaMemcpy(d_A, input.data(), matrix_elems * sizeof(cuComplex),
                          cudaMemcpyHostToDevice));
    CUSOLVER_CHECK(cusolverDnCheevjBatched(handle, CUSOLVER_EIG_MODE_VECTOR,
                                           CUBLAS_FILL_MODE_UPPER, n, d_A, n, d_W,
                                           d_work, lwork, d_info, params, batch_size));
  };

  for (int i = 0; i < warmup; ++i) run_once();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < timed_runs; ++i) run_once();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  SolverResult r;
  r.avg_ms = ms / timed_runs;
  r.eigenvalues_first_matrix.resize(n);
  CUDA_CHECK(cudaMemcpy(r.eigenvalues_first_matrix.data(), d_W, n * sizeof(float),
                       cudaMemcpyDeviceToHost));

  cudaFree(d_A);
  cudaFree(d_W);
  cudaFree(d_info);
  cudaFree(d_work);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cusolverDnDestroySyevjInfo(params);
  cusolverDnDestroy(handle);
  return r;
}

// Truncated eigendecomposition: cusolverDnXsyevdx computes only the top K
// eigenpairs (index range [n-k+1, n], 1-based -- cuSOLVER returns eigenvalues
// ascending, so the top K are the K largest indices) via divide-and-conquer,
// instead of all n. NOT a batched API -- one host-driven call per matrix, so
// this tests whether doing less work per matrix (K of n instead of all n)
// outweighs per-call dispatch overhead repeated batch_size times.
SolverResult bench_syevdx_topk(int n, int k, int batch_size,
                               const std::vector<cuComplex> &input, int warmup,
                               int timed_runs) {
  cusolverDnHandle_t handle;
  cusolverDnParams_t params;
  CUSOLVER_CHECK(cusolverDnCreate(&handle));
  CUSOLVER_CHECK(cusolverDnCreateParams(&params));

  cuComplex *d_A;
  float *d_W; // k eigenvalues per matrix
  int *d_info;
  const size_t matrix_elems = static_cast<size_t>(n) * n * batch_size;
  CUDA_CHECK(cudaMalloc(&d_A, matrix_elems * sizeof(cuComplex)));
  CUDA_CHECK(cudaMalloc(&d_W, static_cast<size_t>(k) * batch_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_info, batch_size * sizeof(int)));

  const int64_t il = n - k + 1, iu = n; // 1-based inclusive index range, top k
  int64_t h_meig = 0;                   // bufferSize wants host (h_ prefix)
  int64_t *d_meig = nullptr;            // the actual call wants device (no prefix)
  CUDA_CHECK(cudaMalloc(&d_meig, sizeof(int64_t)));
  // vl/vu are unused for range=INDEX per the docs, but pass real (dummy)
  // storage rather than nullptr in case the implementation dereferences them
  // unconditionally regardless of range mode.
  float vl_val = 0.0f, vu_val = 0.0f;
  size_t work_device_size = 0, work_host_size = 0;
  // Buffer size is identical for every matrix (same n/k/range), so query once.
  CUSOLVER_CHECK(cusolverDnXsyevdx_bufferSize(
      handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I,
      CUBLAS_FILL_MODE_UPPER, n, CUDA_C_32F, d_A, n, &vl_val, &vu_val, il, iu,
      &h_meig, CUDA_R_32F, d_W, CUDA_C_32F, &work_device_size, &work_host_size));
  void *d_work = nullptr;
  CUDA_CHECK(cudaMalloc(&d_work, work_device_size));
  std::vector<uint8_t> h_work(work_host_size);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto run_once = [&]() {
    CUDA_CHECK(cudaMemcpy(d_A, input.data(), matrix_elems * sizeof(cuComplex),
                          cudaMemcpyHostToDevice));
    for (int b = 0; b < batch_size; ++b) {
      CUSOLVER_CHECK(cusolverDnXsyevdx(
          handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I,
          CUBLAS_FILL_MODE_UPPER, n, CUDA_C_32F,
          d_A + static_cast<size_t>(b) * n * n, n, &vl_val, &vu_val, il, iu,
          d_meig, CUDA_R_32F, d_W + static_cast<size_t>(b) * k, CUDA_C_32F,
          d_work, work_device_size, h_work.data(), work_host_size, d_info));
    }
  };

  for (int i = 0; i < warmup; ++i) run_once();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < timed_runs; ++i) run_once();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  SolverResult r;
  r.avg_ms = ms / timed_runs;
  r.eigenvalues_first_matrix.resize(k);
  CUDA_CHECK(cudaMemcpy(r.eigenvalues_first_matrix.data(), d_W, k * sizeof(float),
                       cudaMemcpyDeviceToHost));

  cudaFree(d_A);
  cudaFree(d_W);
  cudaFree(d_info);
  cudaFree(d_work);
  cudaFree(d_meig);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  return r;
}

// Custom batched top-K solver (batched_topk_eigensolver.cuh): blocked
// subspace iteration + Rayleigh-Ritz, one block per matrix. Returns the K
// eigenvalues/eigenvectors of matrix 0 for the residual check below in
// addition to timing.
struct TopKResult {
  double avg_ms = 0.0;
  std::vector<float> eigenvalues_first_matrix;   // k values, ascending
  std::vector<cuComplex> eigenvectors_first_matrix; // n*k, column-major
};

// oversample: run the kernel with k_work = k + oversample columns instead of
// exactly k, keeping only the top k (largest) of the result. Standard
// subspace-iteration trick: convergence rate is set by the eigenvalue gap at
// the *edge of the working subspace*, not at k -- a few extra columns push
// that edge past a near-degenerate cluster (e.g. a flat noise floor) into a
// region with a real gap, converging much faster for comparable iteration
// counts. Costs more per iteration (bigger Q/AQ/H), so it's a real trade,
// not a free win -- that's exactly what this benchmark is for.
TopKResult bench_custom_topk(int n, int k, int batch_size,
                             const std::vector<cuComplex> &input, int warmup,
                             int timed_runs, int iterations, int oversample) {
  const int k_work = k + oversample;
  cuComplex *d_A;
  float *d_eigvals;
  cuComplex *d_eigvecs;
  const size_t matrix_elems = static_cast<size_t>(n) * n * batch_size;
  CUDA_CHECK(cudaMalloc(&d_A, matrix_elems * sizeof(cuComplex)));
  CUDA_CHECK(cudaMalloc(&d_eigvals, static_cast<size_t>(k_work) * batch_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_eigvecs,
                       static_cast<size_t>(n) * k_work * batch_size * sizeof(cuComplex)));
  CUDA_CHECK(cudaMemcpy(d_A, input.data(), matrix_elems * sizeof(cuComplex),
                       cudaMemcpyHostToDevice));

  const int block_dim = 256;
  const size_t smem = batched_topk::shared_mem_bytes(n, k_work);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto run_once = [&](unsigned seed) {
    batched_topk::batched_topk_kernel<<<batch_size, block_dim, smem>>>(
        d_A, d_eigvals, d_eigvecs, n, k_work, iterations, seed);
  };

  for (int i = 0; i < warmup; ++i) run_once(1u);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < timed_runs; ++i) run_once(1u);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // Kernel writes k_work columns ascending; keep only the top k (the tail).
  std::vector<float> all_eigvals(k_work);
  std::vector<cuComplex> all_eigvecs(static_cast<size_t>(n) * k_work);
  CUDA_CHECK(cudaMemcpy(all_eigvals.data(), d_eigvals, k_work * sizeof(float),
                       cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(all_eigvecs.data(), d_eigvecs,
                       static_cast<size_t>(n) * k_work * sizeof(cuComplex),
                       cudaMemcpyDeviceToHost));

  TopKResult r;
  r.avg_ms = ms / timed_runs;
  r.eigenvalues_first_matrix.assign(all_eigvals.end() - k, all_eigvals.end());
  r.eigenvectors_first_matrix.assign(all_eigvecs.begin() + static_cast<size_t>(n) * oversample,
                                     all_eigvecs.end());

  cudaFree(d_A);
  cudaFree(d_eigvals);
  cudaFree(d_eigvecs);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return r;
}

// Direct correctness check, independent of any reference solver's sign/phase
// conventions: ||A*v - lambda*v|| / ||v|| for each computed eigenpair, using
// matrix 0's *original* (untouched) data. Near zero means (lambda, v)
// genuinely is an eigenpair of A, regardless of whether it matches Xsyev's
// output convention.
double max_residual(int n, int k, const std::vector<cuComplex> &A_matrix0,
                    const std::vector<float> &eigvals, const std::vector<cuComplex> &eigvecs) {
  double worst = 0.0;
  for (int c = 0; c < k; ++c) {
    double resid_norm2 = 0.0, vec_norm2 = 0.0;
    for (int i = 0; i < n; ++i) {
      cuComplex Av_i = make_cuComplex(0.f, 0.f);
      for (int j = 0; j < n; ++j) {
        Av_i = cuCaddf(Av_i, cuCmulf(A_matrix0[i + j * n], eigvecs[j + c * n]));
      }
      const cuComplex lv_i =
          cuCmulf(make_cuComplex(eigvals[c], 0.f), eigvecs[i + c * n]);
      const cuComplex r = cuCsubf(Av_i, lv_i);
      resid_norm2 += static_cast<double>(r.x) * r.x + static_cast<double>(r.y) * r.y;
      const cuComplex v_i = eigvecs[i + c * n];
      vec_norm2 += static_cast<double>(v_i.x) * v_i.x + static_cast<double>(v_i.y) * v_i.y;
    }
    const double rel_resid = std::sqrt(resid_norm2) / std::max(std::sqrt(vec_norm2), 1e-20);
    worst = std::max(worst, rel_resid);
  }
  return worst;
}

double max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
  // Both solvers return ascending-sorted eigenvalues for a valid Hermitian
  // input, so a direct elementwise compare is valid without re-sorting.
  double worst = 0.0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    worst = std::max(worst, static_cast<double>(std::fabs(a[i] - b[i])));
  }
  return worst;
}

void run_comparison(const char *label, int n, int batch_size, int warmup,
                    int timed_runs, double tolerance, int max_sweeps, int k,
                    int topk_timed_runs, bool run_jacobi, bool run_syevdx,
                    int subspace_iterations, int num_sources, float noise_floor,
                    int oversample) {
  auto input = make_psd_visibility_batch(n, batch_size, num_sources, noise_floor, /*seed=*/42);

  auto xsyev = bench_xsyev(n, batch_size, input, warmup, timed_runs);
  std::cout << "--- " << label << " (n=" << n << " batch=" << batch_size << ") ---\n"
            << "  Xsyev        (current, full n=" << n << "):  " << xsyev.avg_ms << " ms/call\n";
  // xsyev's largest k eigenvalues (its tail, ascending) is the reference for
  // both topk candidates below, which return only the top k.
  std::vector<float> xsyev_top_k(xsyev.eigenvalues_first_matrix.end() - k,
                                 xsyev.eigenvalues_first_matrix.end());

  if (run_jacobi) {
    auto cheevj = bench_cheevj(n, batch_size, input, warmup, timed_runs, tolerance, max_sweeps);
    const double speedup = xsyev.avg_ms / cheevj.avg_ms;
    const double eig_diff = max_abs_diff(xsyev.eigenvalues_first_matrix,
                                         cheevj.eigenvalues_first_matrix);
    std::cout << "  Cheevj       (batched Jacobi, tol=" << (tolerance > 0 ? tolerance : 1e-7)
              << " max_sweeps=" << (max_sweeps > 0 ? max_sweeps : 100) << "): "
              << cheevj.avg_ms << " ms/call  => " << speedup << "x"
              << "  | max |eigval diff| = " << eig_diff << "\n";
  }

  if (run_syevdx) {
    auto topk = bench_syevdx_topk(n, k, batch_size, input, warmup, topk_timed_runs);
    const double topk_speedup = xsyev.avg_ms / topk.avg_ms;
    const double topk_diff = max_abs_diff(xsyev_top_k, topk.eigenvalues_first_matrix);
    std::cout << "  Xsyevdx-topk (looped, unbatched, k=" << k << " of " << n << "): "
              << topk.avg_ms << " ms/call  => " << topk_speedup << "x"
              << "  | max |eigval diff| (top " << k << ") = " << topk_diff << "\n";
  }

  auto custom = bench_custom_topk(n, k, batch_size, input, warmup, timed_runs,
                                  subspace_iterations, oversample);
  const double custom_speedup = xsyev.avg_ms / custom.avg_ms;
  const double custom_eig_diff = max_abs_diff(xsyev_top_k, custom.eigenvalues_first_matrix);
  std::vector<cuComplex> A_matrix0(input.begin(), input.begin() + static_cast<size_t>(n) * n);
  const double custom_residual = max_residual(n, k, A_matrix0, custom.eigenvalues_first_matrix,
                                              custom.eigenvectors_first_matrix);
  std::cout << "  Custom-topk  (batched subspace iteration, k=" << k << "+" << oversample
            << " of " << n << ", " << subspace_iterations << " iters): " << custom.avg_ms
            << " ms/call  => " << custom_speedup << "x"
            << "  | max |eigval diff| (top " << k << ") = " << custom_eig_diff
            << "  | max residual ||Av-lv||/||v|| = " << custom_residual << "\n\n";
}

} // namespace

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("bench_eigensolver");
  program.add_argument("--warmup").default_value(5).scan<'i', int>();
  program.add_argument("--timed-runs").default_value(50).scan<'i', int>();
  program.add_argument("--tolerance")
      .help("Jacobi convergence tolerance (0 = cuSOLVER default, ~1e-7 for float)")
      .default_value(0.0)
      .scan<'g', double>();
  program.add_argument("--max-sweeps")
      .help("Jacobi max sweeps (0 = cuSOLVER default, 100)")
      .default_value(0)
      .scan<'i', int>();
  program.add_argument("--n")
      .help("Matrix dimension (NR_RECEIVERS in the real pipeline)")
      .default_value(40)
      .scan<'i', int>();
  program.add_argument("--k")
      .help("Top-K eigenpairs to compute for the truncated (Xsyevdx) candidate")
      .default_value(10)
      .scan<'i', int>();
  program.add_argument("--topk-timed-runs")
      .help("Timed runs for the Xsyevdx-topk candidate specifically (it's an "
            "unbatched per-matrix loop, so large batches take much longer per "
            "run than the batched candidates -- default lower than --timed-runs)")
      .default_value(5)
      .scan<'i', int>();
  program.add_argument("--skip-jacobi")
      .help("Skip the Cheevj (batched Jacobi) candidate -- already ruled out, "
            "skip it to iterate faster on the other candidates")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--run-syevdx")
      .help("Also run the Xsyevdx-topk candidate -- segfaults inside cuSOLVER "
            "itself for this input shape in this cuSOLVER build (see the "
            "investigation notes); off by default")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--subspace-iterations")
      .help("Subspace-iteration count for the custom batched top-K solver "
            "(batched_topk_eigensolver.cuh)")
      .default_value(20)
      .scan<'i', int>();
  program.add_argument("--num-sources")
      .help("Number of dominant 'source' outer-products in the synthetic PSD "
            "visibility matrix (see make_psd_visibility_batch) -- models a "
            "few strong RFI/signal eigenmodes, like the real pipeline's data")
      .default_value(3)
      .scan<'i', int>();
  program.add_argument("--noise-floor")
      .help("Noise-floor eigenvalue added to every mode in the synthetic PSD "
            "matrix")
      .default_value(0.1f)
      .scan<'g', float>();
  program.add_argument("--oversample")
      .help("Extra working-subspace columns beyond k for the custom solver "
            "(k_work=k+oversample; keeps only the top k) -- pushes the "
            "convergence-limiting eigenvalue gap past a near-degenerate "
            "cluster (e.g. a flat noise floor) for faster convergence")
      .default_value(0)
      .scan<'i', int>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << "\n" << program;
    return 1;
  }

  const int warmup = program.get<int>("--warmup");
  const int timed_runs = program.get<int>("--timed-runs");
  const double tolerance = program.get<double>("--tolerance");
  const int max_sweeps = program.get<int>("--max-sweeps");
  const int n = program.get<int>("--n");
  const int k = program.get<int>("--k");
  const int topk_timed_runs = program.get<int>("--topk-timed-runs");
  const bool run_jacobi = !program.get<bool>("--skip-jacobi");
  const bool run_syevdx = program.get<bool>("--run-syevdx");
  const int subspace_iterations = program.get<int>("--subspace-iterations");
  const int num_sources = program.get<int>("--num-sources");
  const float noise_floor = program.get<float>("--noise-floor");
  const int oversample = program.get<int>("--oversample");

  std::cout << "bench_eigensolver: cusolverDnXsyevBatched vs cusolverDnCheevjBatched vs "
               "cusolverDnXsyevdx(top-k) vs custom batched top-k\n"
            << "  n=" << n << " k=" << k << " oversample=" << oversample << " warmup=" << warmup
            << " timed_runs=" << timed_runs << " topk_timed_runs=" << topk_timed_runs
            << " subspace_iterations=" << subspace_iterations
            << " num_sources=" << num_sources << " noise_floor=" << noise_floor
            << "\n  (PSD synthetic visibility matrices -- num_sources dominant "
               "eigenmodes over a noise floor, matching real data's structure)\n\n";

  // Batch sizes matching the real pipeline configs already benchmarked
  // (CUSOLVER_BATCH_SIZE = NR_CHANNELS * NR_POLARIZATIONS^2, NR_POLARIZATIONS=2):
  //   8ch  off -> 32,  8ch  fine32 (224 effective) -> 896
  //   16ch off -> 64,  16ch fine32 (448 effective) -> 1792
  run_comparison("8ch4fpga (off)",        n,   32, warmup, timed_runs, tolerance, max_sweeps,
                 k, topk_timed_runs, run_jacobi, run_syevdx, subspace_iterations,
                 num_sources, noise_floor, oversample);
  run_comparison("8ch4fpga_fine32 (on)",  n,  896, warmup, timed_runs, tolerance, max_sweeps,
                 k, topk_timed_runs, run_jacobi, run_syevdx, subspace_iterations,
                 num_sources, noise_floor, oversample);
  run_comparison("16ch4fpga (off)",       n,   64, warmup, timed_runs, tolerance, max_sweeps,
                 k, topk_timed_runs, run_jacobi, run_syevdx, subspace_iterations,
                 num_sources, noise_floor, oversample);
  run_comparison("16ch4fpga_fine32 (on)", n, 1792, warmup, timed_runs, tolerance, max_sweeps,
                 k, topk_timed_runs, run_jacobi, run_syevdx, subspace_iterations,
                 num_sources, noise_floor, oversample);

  return 0;
}
