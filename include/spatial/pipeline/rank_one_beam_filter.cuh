#pragma once

// A Householder QR gives an orthonormal complement of D^-1 s. Matrices are
// column-major throughout, including the reduced covariance passed to cuSOLVER.
__global__ void solar_complement(const cuComplex *signature, const float *diag,
                                 cuComplex *basis, int n) {
  extern __shared__ cuComplex v[];
  const int batch = blockIdx.x;
  if (threadIdx.x != 0)
    return;
  float norm = 0;
  for (int i = 0; i < n; ++i) {
    const auto z = signature[batch * n + i];
    const float d = diag[batch * n + i];
    v[i] = make_cuComplex(z.x / d, z.y / d);
    norm += v[i].x * v[i].x + v[i].y * v[i].y;
  }
  norm = sqrtf(norm);
  for (int i = 0; i < n; ++i)
    v[i] = make_cuComplex(v[i].x / norm, v[i].y / norm);
  const float first = cuCabsf(v[0]);
  const cuComplex phase = first > 0
                              ? make_cuComplex(v[0].x / first, v[0].y / first)
                              : make_cuComplex(1, 0);
  v[0] = cuCaddf(v[0], phase);
  float vv = 0;
  for (int i = 0; i < n; ++i)
    vv += v[i].x * v[i].x + v[i].y * v[i].y;
  for (int j = 1; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      auto z = cuCmulf(v[i], cuConjf(v[j]));
      basis[batch * n * (n - 1) + (j - 1) * n + i] =
          make_cuComplex((i == j ? 1.f : 0.f) - 2 * z.x / vv, -2 * z.y / vv);
    }
}

__global__ void voltage_filter_from_modes(const cuComplex *modes,
                                          const float *diag,
                                          const int32_t *counts,
                                          __half2 *filter, int n, int columns,
                                          const int *solver_info = nullptr,
                                          unsigned *failures = nullptr) {
  const int batch = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * n)
    return;
  const int row = i % n, col = i / n;
  if (solver_info && solver_info[batch] != 0) {
    if (i == 0 && failures)
      atomicAdd(failures, 1u);
    filter[batch * n * n + i] = __floats2half2_rn(row == col ? 1.f : 0.f, 0.f);
    return;
  }
  cuComplex p = make_cuComplex(row == col ? 1.f : 0.f, 0);
  for (int k = columns - counts[batch]; k < columns; ++k) {
    const auto x = modes[batch * n * columns + k * n + row];
    const auto y = modes[batch * n * columns + k * n + col];
    p = cuCsubf(p, cuCmulf(x, cuConjf(y)));
  }
  const float scale = diag[batch * n + row] / diag[batch * n + col];
  filter[batch * n * n + i] = __floats2half2_rn(scale * p.x, scale * p.y);
}

class RankOneBeamFilter {
  int n_, batches_;
  DevicePtr<cuComplex> signature_, basis_, intermediate_, reduced_, modes_;
  DevicePtr<float> values_;
  DevicePtr<int> info_;
  DevicePtr<void> workspace_;
  std::vector<char> host_workspace_;
  size_t device_bytes_ = 0;

public:
  RankOneBeamFilter(int n, int batches,
                    const std::vector<std::complex<float>> &signature,
                    cusolverDnHandle_t solver, cusolverDnParams_t params)
      : n_(n), batches_(batches),
        signature_(make_device_ptr<cuComplex>(batches * n * sizeof(cuComplex))),
        basis_(make_device_ptr<cuComplex>(batches * n * (n - 1) *
                                          sizeof(cuComplex))),
        intermediate_(make_device_ptr<cuComplex>(batches * n * (n - 1) *
                                                 sizeof(cuComplex))),
        reduced_(make_device_ptr<cuComplex>(batches * (n - 1) * (n - 1) *
                                            sizeof(cuComplex))),
        modes_(make_device_ptr<cuComplex>(batches * n * (n - 1) *
                                          sizeof(cuComplex))),
        values_(make_device_ptr<float>(batches * (n - 1) * sizeof(float))),
        info_(make_device_ptr<int>(batches * sizeof(int))) {
    CUDA_CHECK(cudaMemcpy(signature_.get(), signature.data(),
                          signature.size() * sizeof(cuComplex),
                          cudaMemcpyHostToDevice));
    size_t host_bytes = 0;
    CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
        solver, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n - 1,
        CUDA_C_32F, reduced_.get(), n - 1, CUDA_R_32F, values_.get(),
        CUDA_C_32F, &device_bytes_, &host_bytes, batches));
    workspace_ = make_device_ptr<void>(device_bytes_);
    host_workspace_.resize(host_bytes);
  }

  void update_signature(const std::vector<std::complex<float>> &signature) {
    CUDA_CHECK(cudaMemcpy(signature_.get(), signature.data(),
                          signature.size() * sizeof(cuComplex),
                          cudaMemcpyHostToDevice));
  }

  void run(cuComplex *covariance, const float *diag, const int32_t *counts,
           __half2 *filter, cudaStream_t stream, cublasHandle_t blas,
           cusolverDnHandle_t solver, cusolverDnParams_t params,
           unsigned *failures) {
    const int n = n_, m = n - 1;
    solar_complement<<<batches_, 1, n * sizeof(cuComplex), stream>>>(
        signature_.get(), diag, basis_.get(), n);
    CUDA_CHECK(cudaGetLastError());
    const cuComplex one{1, 0}, zero{0, 0};
    CUBLAS_CHECK(cublasCgemmStridedBatched(
        blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &one, covariance, n, n * n,
        basis_.get(), n, n * m, &zero, intermediate_.get(), n, n * m,
        batches_));
    CUBLAS_CHECK(cublasCgemmStridedBatched(blas, CUBLAS_OP_C, CUBLAS_OP_N, m, m,
                                           n, &one, basis_.get(), n, n * m,
                                           intermediate_.get(), n, n * m, &zero,
                                           reduced_.get(), m, m * m, batches_));
    CUSOLVER_CHECK(cusolverDnXsyevBatched(
        solver, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, m,
        CUDA_C_32F, reduced_.get(), m, CUDA_R_32F, values_.get(), CUDA_C_32F,
        workspace_.get(), device_bytes_, host_workspace_.data(),
        host_workspace_.size(), info_.get(), batches_));
    CUBLAS_CHECK(cublasCgemmStridedBatched(
        blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, &one, basis_.get(), n, n * m,
        reduced_.get(), m, m * m, &zero, modes_.get(), n, n * m, batches_));
    voltage_filter_from_modes<<<dim3((n * n + 255) / 256, batches_), 256, 0,
                                stream>>>(modes_.get(), diag, counts, filter, n,
                                          m, info_.get(), failures);
    CUDA_CHECK(cudaGetLastError());
  }
};
