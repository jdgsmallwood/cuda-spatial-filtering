#include "spatial/spatial.hpp"
#include "ccglib/common/precision.h"
#include <ccglib/ccglib.hpp>
#include <ccglib/common/complex_order.h>
#include <ccglib/transpose/transpose.h>
#include <complex>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudawrappers/cu.hpp>
#include <cusolverDn.h>
#include <iostream>
#include <libtcc/Correlator.h>
#include <vector>

template <typename T>
void eigendecomposition(float *h_eigenvalues, int n, const std::vector<T> *A) {
  T *d_A;
  cudaMalloc((void **)&d_A, n * n * sizeof(T));
  cudaMemcpy(d_A, A->data(), sizeof(T) * n * n, cudaMemcpyHostToDevice);

  //// Allocate memory for eigenvalues and eigenvectors
  float *d_eigenvalues;
  T *d_eigenvectors;
  cudaMalloc((void **)&d_eigenvalues, n * sizeof(float));
  cudaMalloc((void **)&d_eigenvectors, n * n * sizeof(T));

  // float* h_eigenvalues;
  // cudaMallocHost((void**)&h_eigenvalues, n * sizeof(float));

  //
  //// Create cuSOLVER handle
  cusolverDnHandle_t solverHandle;
  cusolverDnCreate(&solverHandle);
  //
  //// Compute eigenvalues and eigenvectors
  int *d_info = nullptr;
  int info = 0;
  cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));

  cudaDataType data_type = CUDA_C_32F;

  cusolverDnParams_t params = NULL;
  cusolverDnCreateParams(&params);

  size_t workspaceInBytesOnDevice = 0;
  size_t workspaceInBytesOnHost = 0;
  void *d_work = nullptr;
  void *h_work = nullptr;

  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  cusolverDnXsyevd_bufferSize(solverHandle, params, jobz, uplo, n, data_type,
                              d_A, n, CUDA_R_32F, d_eigenvalues, data_type,
                              &workspaceInBytesOnDevice,
                              &workspaceInBytesOnHost);

  cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice);
  h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));

  cusolverDnXsyevd(
      solverHandle, params,
      jobz, // Mode - CUSOLVER_EIG_MODE_VECTOR = get eigenvalues & eigenvectors
      uplo, // cublasFillMode_t
      n,    // size of symmetric matrix
      data_type,  // data type
      d_A,        // what to decompose
      n,          // lda
      CUDA_R_32F, // data type output - should always be real for the eigenvalue
                  // outputs.
      d_eigenvalues, // array to store eigenvalues
      data_type,     // data type of computation
      d_work, workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost,
      d_info); //

  //// Check for errors
  int h_info;
  cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  if (h_info != 0) {
    std::cerr << "Eigenvalue decomposition failed!" << std::endl;
    return;
  }

  cudaMemcpy(h_eigenvalues, d_eigenvalues, n * sizeof(float),
             cudaMemcpyDeviceToHost);
  //

  printf("Eigenvalues...\n");
  for (int i = 0; i < n; i++) {
    printf("%f \n", h_eigenvalues[i]);
  }

  //// Destroy cuSOLVER handle
  cusolverDnDestroy(solverHandle);
}

template void eigendecomposition<cuComplex>(float *h_eigenvalues, int n,
                                            const std::vector<cuComplex> *A);

template <typename T>
void d_eigendecomposition(float *d_eigenvalues, const int n, T *d_A,
                          cudaStream_t stream) {
  //// Create cuSOLVER handle
  cusolverDnHandle_t solverHandle;
  cusolverDnCreate(&solverHandle);
  //
  //// Compute eigenvalues and eigenvectors
  int *d_info = nullptr;
  int info = 0;
  cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));

  cudaDataType data_type = CUDA_C_32F;

  cusolverDnParams_t params = NULL;
  cusolverDnCreateParams(&params);

  size_t workspaceInBytesOnDevice = 0;
  size_t workspaceInBytesOnHost = 0;
  void *d_work = nullptr;
  void *h_work = nullptr;

  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  cusolverDnXsyevd_bufferSize(solverHandle, params, jobz, uplo, n, data_type,
                              d_A, n, CUDA_R_32F, d_eigenvalues, data_type,
                              &workspaceInBytesOnDevice,
                              &workspaceInBytesOnHost);

  cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice);
  h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));

  cusolverDnSetStream(solverHandle, stream);
  cusolverDnXsyevd(
      solverHandle, params,
      jobz, // Mode - CUSOLVER_EIG_MODE_VECTOR = get eigenvalues & eigenvectors
      uplo, // cublasFillMode_t
      n,    // size of symmetric matrix
      data_type,  // data type
      d_A,        // what to decompose
      n,          // lda
      CUDA_R_32F, // data type output - should always be real for the eigenvalue
                  // outputs.
      d_eigenvalues, // array to store eigenvalues
      data_type,     // data type of computation
      d_work, workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost,
      d_info); //

  //// Check for errors
  int h_info;
  cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  if (h_info != 0) {
    std::cerr << "Eigenvalue decomposition failed!" << std::endl;
    return;
  }
  //// Destroy cuSOLVER handle
  cusolverDnDestroy(solverHandle);
}

template void eigendecomposition<cuComplex>(float *h_eigenvalues, int n,
                                            const std::vector<cuComplex> *A);

void correlate(Samples *samples, Visibilities *visibilities) {
  try {
    // Taken from simpleExample
    std::cout << "Starting correlation inline" << std::endl;
    checkCudaCall(
        cudaSetDevice(0)); // combine the CUDA runtime API and CUDA driver API
    checkCudaCall(cudaFree(0));
    std::cout << "Instantiating correlator..." << std::endl;
    tcc::Correlator correlator(cu::Device(0), inputFormat, NR_RECEIVERS,
                               NR_CHANNELS, NR_SAMPLES_PER_CHANNEL,
                               NR_POLARIZATIONS, NR_RECEIVERS_PER_BLOCK);

    cudaStream_t stream;
    checkCudaCall(cudaStreamCreate(&stream));

    Samples *d_samples;
    Visibilities *d_visibilities;
    checkCudaCall(cudaMalloc(&d_samples, sizeof(Samples)));
    checkCudaCall(cudaMalloc(&d_visibilities, sizeof(Visibilities)));

    checkCudaCall(cudaMemcpyAsync(d_samples, samples, sizeof(Samples),
                                  cudaMemcpyHostToDevice, stream));

    std::cout << "Starting correlator" << std::endl;
    correlator.launchAsync((CUstream)stream, (CUdeviceptr)d_visibilities,
                           (CUdeviceptr)d_samples);
    checkCudaCall(cudaMemcpyAsync(visibilities, d_visibilities,
                                  sizeof(Visibilities), cudaMemcpyDeviceToHost,
                                  stream));
    std::cout << "Synchronizing..." << std::endl;
    checkCudaCall(cudaStreamSynchronize(stream));
    std::cout << "Synchronized" << std::endl;

    cudaFree(d_samples);
    cudaFree(d_visibilities);

    checkCudaCall(cudaStreamDestroy(stream));
  } catch (std::exception &error) {
    std::cerr << error.what() << std::endl;
  }
}

template <typename T, typename S>
void beamform(std::complex<T> *data_matrix, std::complex<T> *weights,
              std::complex<S> *output_matrix, const int n_antennas,
              const int n_samples, const int n_beams) {
  T *data_matrix_reshaped, *weights_reshaped, *d_data, *d_weights;
  S *output_reshaped, *d_output;

  cudaMallocHost((void **)&data_matrix_reshaped,
                 n_antennas * n_samples * sizeof(T) * 2);
  cudaMallocHost((void **)&weights_reshaped,
                 n_beams * n_antennas * sizeof(T) * 2);
  cudaMallocHost((void **)&output_reshaped,
                 n_beams * n_samples * sizeof(S) * 2);

  rearrange_matrix_to_ccglib_format<__half>(data_matrix, data_matrix_reshaped,
                                            n_antennas, n_samples, true);
  rearrange_matrix_to_ccglib_format<__half>(weights, weights_reshaped, n_beams,
                                            n_antennas, true);

  cudaMalloc((void **)&d_data, n_antennas * n_samples * 2 * sizeof(T));
  cudaMalloc((void **)&d_weights, n_beams * n_antennas * 2 * sizeof(T));
  cudaMalloc((void **)&d_output, n_beams * n_samples * 2 * sizeof(S));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaMemcpyAsync(d_data, data_matrix_reshaped,
                  n_antennas * n_samples * 2 * sizeof(T), cudaMemcpyDefault,
                  stream);
  cudaMemcpyAsync(d_weights, weights_reshaped,
                  n_beams * n_antennas * 2 * sizeof(T), cudaMemcpyDefault,
                  stream);

  CUdevice cu_device;
  cuDeviceGet(&cu_device, 0);
  ccglib::mma::GEMM gemm_mma(1, n_beams, n_samples, n_antennas, cu_device,
                             stream, ccglib::ValueType::float16,
                             ccglib::mma::basic);

  gemm_mma.Run((CUdeviceptr)d_weights, (CUdeviceptr)d_data,
               (CUdeviceptr)d_output);

  cudaMemcpyAsync(output_reshaped, d_output,
                  n_beams * n_samples * 2 * sizeof(S), cudaMemcpyDefault,
                  stream);

  cudaStreamSynchronize(stream);

  rearrange_ccglib_matrix_to_compact_format(output_reshaped, output_matrix,
                                            n_beams, n_samples);

  cudaStreamDestroy(stream);
  cudaFreeHost(data_matrix_reshaped);
  cudaFreeHost(weights_reshaped);
  cudaFreeHost(output_reshaped);

  cudaFree(d_output);
  cudaFree(d_weights);
  cudaFree(d_data);
}

template void beamform(std::complex<__half> *data_matrix,
                       std::complex<__half> *weights,
                       std::complex<float> *output_matrix, const int n_antennas,
                       const int n_samples, const int n_beams);

void ccglib_mma(__half *A, __half *B, float *C, const int n_row,
                const int n_col, const int batch_size, int n_inner) {
  if (n_inner == -1) {
    n_inner = n_row;
  }

  // The format of A is n_row x n_col of real parts of matrix and then n_row x
  // n_col of imag parts of matrix.
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

  __half *d_A, *d_B;
  checkCudaCall(
      cudaMalloc(&d_A, sizeof(__half) * 2 * n_row * n_inner * batch_size));

  checkCudaCall(
      cudaMalloc(&d_B, sizeof(__half) * 2 * n_inner * n_col * batch_size));

  float(*d_C);

  checkCudaCall(
      cudaMalloc(&d_C, sizeof(float) * 2 * n_row * n_col * batch_size));
  cudaStream_t stream;
  checkCudaCall(cudaStreamCreate(&stream));
  checkCudaCall(
      cudaMemcpyAsync(d_A, A, sizeof(__half) * 2 * n_row * n_inner * batch_size,
                      cudaMemcpyHostToDevice, stream));
  checkCudaCall(
      cudaMemcpyAsync(d_B, B, sizeof(__half) * 2 * n_inner * n_col * batch_size,
                      cudaMemcpyHostToDevice, stream));

  CUdevice cu_device;
  cuDeviceGet(&cu_device, 0);
  ccglib::mma::GEMM gemm_mma(batch_size, n_row, n_col, n_inner, cu_device,
                             stream, ccglib::ValueType::float16,
                             ccglib::mma::basic);

  gemm_mma.Run((CUdeviceptr)d_A, (CUdeviceptr)d_B, (CUdeviceptr)d_C);
  checkCudaCall(cudaMemcpyAsync(C, d_C,
                                sizeof(float) * 2 * n_row * n_col * batch_size,
                                cudaMemcpyDeviceToHost, stream));

  checkCudaCall(cudaStreamSynchronize(stream));

  checkCudaCall(cudaFree(d_A));
  checkCudaCall(cudaFree(d_B));
  checkCudaCall(cudaFree(d_C));
}

void ccglib_mma_opt(__half *A, __half *B, float *C, const int n_row,
                    const int n_col, const int batch_size, int n_inner,
                    const int tile_size_x, const int tile_size_y) {
  if (n_inner == -1) {
    n_inner = n_row;
  }

  // The format of A is n_row x n_col of real parts of matrix and then n_row x
  // n_col of imag parts of matrix.
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

  __half *d_A, *d_B, *d_A_T, *d_B_T;
  checkCudaCall(
      cudaMalloc(&d_A, sizeof(__half) * 2 * n_row * n_inner * batch_size));
  checkCudaCall(
      cudaMalloc(&d_B, sizeof(__half) * 2 * n_inner * n_col * batch_size));
  checkCudaCall(
      cudaMalloc(&d_A_T, sizeof(__half) * 2 * n_row * n_inner * batch_size));
  checkCudaCall(
      cudaMalloc(&d_B_T, sizeof(__half) * 2 * n_inner * n_col * batch_size));
  float *d_C;

  checkCudaCall(
      cudaMalloc(&d_C, sizeof(float) * 2 * n_row * n_col * batch_size));
  cudaStream_t stream;
  checkCudaCall(cudaStreamCreate(&stream));
  checkCudaCall(
      cudaMemcpyAsync(d_A, A, sizeof(__half) * 2 * n_row * n_inner * batch_size,
                      cudaMemcpyHostToDevice, stream));
  checkCudaCall(
      cudaMemcpyAsync(d_B, B, sizeof(__half) * 2 * n_inner * n_col * batch_size,
                      cudaMemcpyHostToDevice, stream));

  CUdevice cu_device;
  cuDeviceGet(&cu_device, 0);

  ccglib::transpose::Transpose transpose_A(
      batch_size, n_row, n_inner, tile_size_x, tile_size_y,
      ccglib::Precision(ccglib::ValueType::float16).GetInputBits(), cu_device,
      stream, ccglib::ComplexAxisLocation::complex_planar);

  transpose_A.Run((CUdeviceptr)d_A, (CUdeviceptr)d_A_T);

  ccglib::transpose::Transpose transpose_B(
      batch_size, n_inner, n_col, tile_size_x, tile_size_y,
      ccglib::Precision(ccglib::ValueType::float16).GetInputBits(), cu_device,
      stream, ccglib::ComplexAxisLocation::complex_planar);

  transpose_B.Run((CUdeviceptr)d_B, (CUdeviceptr)d_B_T);

  ccglib::mma::GEMM gemm_mma(batch_size, n_row, n_col, n_inner, cu_device,
                             stream, ccglib::ValueType::float16,
                             ccglib::mma::opt);

  gemm_mma.Run((CUdeviceptr)d_A_T, (CUdeviceptr)d_B_T, (CUdeviceptr)d_C);
  checkCudaCall(cudaMemcpyAsync(C, d_C,
                                sizeof(float) * 2 * n_row * n_col * batch_size,
                                cudaMemcpyDeviceToHost, stream));

  checkCudaCall(cudaStreamSynchronize(stream));
  cudaStreamDestroy(stream);
  checkCudaCall(cudaFree(d_A));
  checkCudaCall(cudaFree(d_B));
  checkCudaCall(cudaFree(d_C));
}

template <typename T>
void rearrange_matrix_to_ccglib_format(const std::complex<T> *input_matrix,
                                       T *output_matrix, const int n_row,
                                       const int n_col, const bool row_major) {
  /*
   * This function will convert a matrix that is in form of std::complex<T> to
   * the form where all the real parts are contiguous in memory and all the
   * imaginary parts are contiguous in memory.
   * */
  int n_major, n_minor;
  if (row_major) {
    n_major = n_row;
    n_minor = n_col;

  } else {
    n_major = n_col;
    n_minor = n_row;
  }
  const int total_elements = n_major * n_minor;
  int idx;
  for (int maj = 0; maj < n_major; ++maj) {
    for (int min = 0; min < n_minor; ++min) {
      idx = maj * n_minor + min;
      std::complex<T> val = input_matrix[idx];
      output_matrix[idx] = val.real();
      output_matrix[total_elements + idx] = val.imag();
    }
  }
}

template void
rearrange_matrix_to_ccglib_format(const std::complex<__half> *input_matrix,
                                  __half *output_matrix, const int n_rows,
                                  const int n_cols, const bool row_major);

template <typename T>
void rearrange_ccglib_matrix_to_compact_format(const T *input_matrix,
                                               std::complex<T> *output_matrix,
                                               const int n_rows,
                                               const int n_cols) {

  const int total_elements = n_rows * n_cols;

  for (int i = 0; i < n_rows; ++i) {
    for (int j = 0; j < n_cols; ++j) {
      output_matrix[i * n_cols + j] =
          std::complex<T>(input_matrix[i * n_cols + j],
                          input_matrix[total_elements + i * n_cols + j]);
    }
  }
}

template void
rearrange_ccglib_matrix_to_compact_format(const float *input_matrix,
                                          std::complex<float> *output_matrix,
                                          const int n_row, const int n_col);
