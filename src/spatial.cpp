#include "spatial/spatial.hpp"
#include "ccglib/common/precision.h"
#include "ccglib/common/value_type.h"
#include "ccglib/gemm/mma.h"
#include "ccglib/gemm/variant.h"
#include "spatial/ethernet.hpp"
#include "spatial/logging.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/tensor.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <ccglib/ccglib.hpp>
#include <ccglib/common/complex_order.h>
#include <ccglib/transpose/transpose.h>
#include <complex>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudawrappers/cu.hpp>
#include <cusolverDn.h>
#include <cusolver_common.h>
#include <exception>
#include <iostream>
#include <library_types.h>
#include <libtcc/Correlator.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <highfive/highfive.hpp>
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
  free(h_work);
}

template void eigendecomposition<cuComplex>(float *h_eigenvalues, int n,
                                            const std::vector<cuComplex> *A);

template <typename T>
void d_eigendecomposition(float *d_eigenvalues, const int n,
                          const int num_channels, const int num_polarizations,
                          T *d_A, cudaStream_t stream) {
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
  for (auto i = 0; i < num_channels; ++i) {
    for (auto j = 0; j < num_polarizations; ++j) {
      cusolverDnXsyevd(
          solverHandle, params,
          jobz,      // Mode - CUSOLVER_EIG_MODE_VECTOR = get
                     // eigenvalues & eigenvectors
          uplo,      // cublasFillMode_t
          n,         // size of symmetric matrix
          data_type, // data type
                     // this is almost certainly not right.
          (T *)&d_A[i * num_polarizations * num_polarizations *
                        spatial::NR_BASELINES +
                    j * n * num_polarizations], // what to decompose
          n,                                    // lda
          CUDA_R_32F, // data type output - should always be real
                      // for the eigenvalue outputs.
          // This should be [CHANNEL][POLARIZATION][EIGENVALUES]
          // This current index is almost certainly not right.
          (float *)&d_eigenvalues[i * n * num_polarizations +
                                  j * n], // array to store eigenvalues
          data_type,                      // data type of computation
          d_work, workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost,
          d_info); //
    }
  }

  //// Check for errors
  int h_info;
  cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  if (h_info != 0) {
    std::cerr << "Eigenvalue decomposition failed!" << std::endl;
    return;
  }
  free(h_work);
  //// Destroy cuSOLVER handle
  cusolverDnDestroy(solverHandle);
}

template void d_eigendecomposition<cuComplex>(float *h_eigenvalues, const int n,
                                              const int num_channels,
                                              const int num_polarizations,
                                              cuComplex *d_A,
                                              cudaStream_t stream);

template void d_eigendecomposition<std::complex<float>>(
    float *h_eigenvalues, const int n, const int num_channels,
    const int num_polarizations, std::complex<float> *d_A, cudaStream_t stream);
void correlate(Samples *samples, Visibilities *visibilities) {
  try {
    // Taken from simpleExample
    LOG_INFO("Starting correlation inline");
    checkCudaCall(
        cudaSetDevice(0)); // combine the CUDA runtime API and CUDA driver API
    checkCudaCall(cudaFree(0));
    LOG_INFO("Instantiating correlator...");
    tcc::Correlator correlator(
        cu::Device(0), inputFormat, NR_RECEIVERS_DEF, NR_CHANNELS_DEF,
        spatial::NR_BLOCKS_FOR_CORRELATION * spatial::NR_TIMES_PER_BLOCK,
        NR_POLARIZATIONS_DEF, NR_RECEIVERS_PER_BLOCK_DEF);

    cudaStream_t stream;
    checkCudaCall(cudaStreamCreate(&stream));

    Samples *d_samples;
    Visibilities *d_visibilities;
    checkCudaCall(cudaMalloc(&d_samples, sizeof(Samples)));
    checkCudaCall(cudaMalloc(&d_visibilities, sizeof(Visibilities)));

    checkCudaCall(cudaMemcpyAsync(d_samples, samples, sizeof(Samples),
                                  cudaMemcpyHostToDevice, stream));

    LOG_INFO("Starting correlator");
    correlator.launchAsync((CUstream)stream, (CUdeviceptr)d_visibilities,
                           (CUdeviceptr)d_samples);
    checkCudaCall(cudaMemcpyAsync(visibilities, d_visibilities,
                                  sizeof(Visibilities), cudaMemcpyDeviceToHost,
                                  stream));
    LOG_INFO("Synchronizing...");
    checkCudaCall(cudaStreamSynchronize(stream));
    LOG_INFO("Synchronized");

    cudaFree(d_samples);
    cudaFree(d_visibilities);

    checkCudaCall(cudaStreamDestroy(stream));
  } catch (std::exception &error) {
    LOG_ERROR("{}", error.what());
  }
}

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
                      cudaMemcpyDefault, stream));
  checkCudaCall(
      cudaMemcpyAsync(d_B, B, sizeof(__half) * 2 * n_inner * n_col * batch_size,
                      cudaMemcpyDefault, stream));

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

KernelSocketPacketCapture::KernelSocketPacketCapture(int port, int buffer_size)
    : port(port), buffer_size(buffer_size) {

  LOG_INFO("UDP Server with concurrent processing starting on port {}...",
           port);
  // Create UDP socket
  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0) {
    perror("socket");
  }
  // Allow address reuse
  int reuse = 1;
  if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
    perror("setsockopt");
  }

  // Setup server address
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = INADDR_ANY;
  server_addr.sin_port = htons(port);

  // Bind socket
  if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
    perror("bind");
    close(sockfd);
  }

  LOG_INFO("Size of PacketPayload is {} bytes...", sizeof(PacketPayload));
  LOG_INFO("Server listening on 0.0.0.0:{}", port);
  LOG_INFO("Press Ctrl+C to stop\n");
}

KernelSocketPacketCapture::~KernelSocketPacketCapture() { close(sockfd); }
