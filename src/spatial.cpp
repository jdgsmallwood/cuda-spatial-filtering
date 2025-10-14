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
// template <typename T>
// void eigendecomposition(float *h_eigenvalues, int n, const std::vector<T> *A)
// {
//   T *d_A;
//   cudaMalloc((void **)&d_A, n * n * sizeof(T));
//   cudaMemcpy(d_A, A->data(), sizeof(T) * n * n, cudaMemcpyHostToDevice);
//
//   //// Allocate memory for eigenvalues and eigenvectors
//   float *d_eigenvalues;
//   T *d_eigenvectors;
//   cudaMalloc((void **)&d_eigenvalues, n * sizeof(float));
//   cudaMalloc((void **)&d_eigenvectors, n * n * sizeof(T));
//
//   // float* h_eigenvalues;
//   // cudaMallocHost((void**)&h_eigenvalues, n * sizeof(float));
//
//   //
//   //// Create cuSOLVER handle
//   cusolverDnHandle_t solverHandle;
//   cusolverDnCreate(&solverHandle);
//   //
//   //// Compute eigenvalues and eigenvectors
//   int *d_info = nullptr;
//   int info = 0;
//   cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
//
//   cudaDataType data_type = CUDA_C_32F;
//
//   cusolverDnParams_t params = NULL;
//   cusolverDnCreateParams(&params);
//
//   size_t workspaceInBytesOnDevice = 0;
//   size_t workspaceInBytesOnHost = 0;
//   void *d_work = nullptr;
//   void *h_work = nullptr;
//
//   cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
//   cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
//
//   cusolverDnXsyevd_bufferSize(solverHandle, params, jobz, uplo, n, data_type,
//                               d_A, n, CUDA_R_32F, d_eigenvalues, data_type,
//                               &workspaceInBytesOnDevice,
//                               &workspaceInBytesOnHost);
//
//   cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice);
//   h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
//
//   cusolverDnXsyevd(
//       solverHandle, params,
//       jobz, // Mode - CUSOLVER_EIG_MODE_VECTOR = get eigenvalues &
//       eigenvectors uplo, // cublasFillMode_t n,    // size of symmetric
//       matrix data_type,  // data type d_A,        // what to decompose n, //
//       lda CUDA_R_32F, // data type output - should always be real for the
//       eigenvalue
//                   // outputs.
//       d_eigenvalues, // array to store eigenvalues
//       data_type,     // data type of computation
//       d_work, workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost,
//       d_info); //
//
//   //// Check for errors
//   int h_info;
//   cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
//   if (h_info != 0) {
//     std::cerr << "Eigenvalue decomposition failed!" << std::endl;
//     return;
//   }
//
//   cudaMemcpy(h_eigenvalues, d_eigenvalues, n * sizeof(float),
//              cudaMemcpyDeviceToHost);
//   //
//
//   printf("Eigenvalues...\n");
//   for (int i = 0; i < n; i++) {
//     printf("%f \n", h_eigenvalues[i]);
//   }
//
//   //// Destroy cuSOLVER handle
//   cusolverDnDestroy(solverHandle);
//   free(h_work);
// }
//
// template void eigendecomposition<cuComplex>(float *h_eigenvalues, int n,
//                                             const std::vector<cuComplex> *A);
//
// template <typename T>
// void d_eigendecomposition(float *d_eigenvalues, const int n,
//                           const int num_channels, const int
//                           num_polarizations, T *d_A, cudaStream_t stream) {
//   //// Create cuSOLVER handle
//   cusolverDnHandle_t solverHandle;
//   cusolverDnCreate(&solverHandle);
//   //
//   //// Compute eigenvalues and eigenvectors
//   int *d_info = nullptr;
//   int info = 0;
//   cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
//
//   cudaDataType data_type = CUDA_C_32F;
//
//   cusolverDnParams_t params = NULL;
//   cusolverDnCreateParams(&params);
//
//   size_t workspaceInBytesOnDevice = 0;
//   size_t workspaceInBytesOnHost = 0;
//   void *d_work = nullptr;
//   void *h_work = nullptr;
//
//   cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
//   cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
//
//   cusolverDnXsyevd_bufferSize(solverHandle, params, jobz, uplo, n, data_type,
//                               d_A, n, CUDA_R_32F, d_eigenvalues, data_type,
//                               &workspaceInBytesOnDevice,
//                               &workspaceInBytesOnHost);
//
//   cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice);
//   h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
//
//   cusolverDnSetStream(solverHandle, stream);
//   for (auto i = 0; i < num_channels; ++i) {
//     for (auto j = 0; j < num_polarizations; ++j) {
//       cusolverDnXsyevd(
//           solverHandle, params,
//           jobz,      // Mode - CUSOLVER_EIG_MODE_VECTOR = get
//                      // eigenvalues & eigenvectors
//           uplo,      // cublasFillMode_t
//           n,         // size of symmetric matrix
//           data_type, // data type
//                      // this is almost certainly not right.
//           (T *)&d_A[i * num_polarizations * num_polarizations *
//                         spatial::NR_BASELINES +
//                     j * n * num_polarizations], // what to decompose
//           n,                                    // lda
//           CUDA_R_32F, // data type output - should always be real
//                       // for the eigenvalue outputs.
//           // This should be [CHANNEL][POLARIZATION][EIGENVALUES]
//           // This current index is almost certainly not right.
//           (float *)&d_eigenvalues[i * n * num_polarizations +
//                                   j * n], // array to store eigenvalues
//           data_type,                      // data type of computation
//           d_work, workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost,
//           d_info); //
//     }
//   }
//
//   //// Check for errors
//   int h_info;
//   cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
//   if (h_info != 0) {
//     std::cerr << "Eigenvalue decomposition failed!" << std::endl;
//     return;
//   }
//   free(h_work);
//   //// Destroy cuSOLVER handle
//   cusolverDnDestroy(solverHandle);
// }
//
// template void d_eigendecomposition<cuComplex>(float *h_eigenvalues, const int
// n,
//                                               const int num_channels,
//                                               const int num_polarizations,
//                                               cuComplex *d_A,
//                                               cudaStream_t stream);
//
// template void d_eigendecomposition<std::complex<float>>(
//     float *h_eigenvalues, const int n, const int num_channels,
//     const int num_polarizations, std::complex<float> *d_A, cudaStream_t
//     stream);

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
  CUDA_CHECK(
      cudaMalloc(&d_A, sizeof(__half) * 2 * n_row * n_inner * batch_size));

  CUDA_CHECK(
      cudaMalloc(&d_B, sizeof(__half) * 2 * n_inner * n_col * batch_size));

  float(*d_C);

  CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * 2 * n_row * n_col * batch_size));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaMemcpyAsync(d_A, A,
                             sizeof(__half) * 2 * n_row * n_inner * batch_size,
                             cudaMemcpyDefault, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, B,
                             sizeof(__half) * 2 * n_inner * n_col * batch_size,
                             cudaMemcpyDefault, stream));

  CUdevice cu_device;
  cuDeviceGet(&cu_device, 0);
  ccglib::mma::GEMM gemm_mma(batch_size, n_row, n_col, n_inner, cu_device,
                             stream, ccglib::ValueType::float16,
                             ccglib::mma::basic);

  gemm_mma.Run((CUdeviceptr)d_A, (CUdeviceptr)d_B, (CUdeviceptr)d_C);
  CUDA_CHECK(cudaMemcpyAsync(C, d_C,
                             sizeof(float) * 2 * n_row * n_col * batch_size,
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
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
  CUDA_CHECK(
      cudaMalloc(&d_A, sizeof(__half) * 2 * n_row * n_inner * batch_size));
  CUDA_CHECK(
      cudaMalloc(&d_B, sizeof(__half) * 2 * n_inner * n_col * batch_size));
  CUDA_CHECK(
      cudaMalloc(&d_A_T, sizeof(__half) * 2 * n_row * n_inner * batch_size));
  CUDA_CHECK(
      cudaMalloc(&d_B_T, sizeof(__half) * 2 * n_inner * n_col * batch_size));
  float *d_C;

  CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * 2 * n_row * n_col * batch_size));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaMemcpyAsync(d_A, A,
                             sizeof(__half) * 2 * n_row * n_inner * batch_size,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, B,
                             sizeof(__half) * 2 * n_inner * n_col * batch_size,
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
  CUDA_CHECK(cudaMemcpyAsync(C, d_C,
                             sizeof(float) * 2 * n_row * n_col * batch_size,
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
}

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
