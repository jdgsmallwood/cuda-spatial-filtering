#include "spatial/spatial.hpp"
#include "ccglib/common/precision.h"
#include "spatial/spatial.cuh"
#include "spatial/tensor.hpp"
#include <atomic>
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
#include <unordered_map>
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
          (T *)&d_A[i * num_polarizations * num_polarizations * NR_BASELINES +
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
    std::cout << "Starting correlation inline" << std::endl;
    checkCudaCall(
        cudaSetDevice(0)); // combine the CUDA runtime API and CUDA driver API
    checkCudaCall(cudaFree(0));
    std::cout << "Instantiating correlator..." << std::endl;
    tcc::Correlator correlator(cu::Device(0), inputFormat, NR_RECEIVERS,
                               NR_CHANNELS,
                               NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK,
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

template <typename T, typename S, typename U>
void beamform(std::complex<T> *h_samples, std::complex<U> *h_weights,
              std::complex<S> *h_beam_output,
              std::complex<S> *h_visibilities_output,
              const int nr_aggregated_packets) {

  constexpr int num_weights =
      NR_BEAMS * NR_RECEIVERS * NR_POLARIZATIONS * NR_CHANNELS;

  constexpr int num_eigen = NR_RECEIVERS * NR_CHANNELS * NR_POLARIZATIONS;
  // create CUDA streams
  cudaStream_t streams[NR_BUFFERS];
  cudaEvent_t input_transfer_done[NR_BUFFERS];

  for (auto i = 0; i < NR_BUFFERS; ++i) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    cudaEventCreate(&input_transfer_done[i]);
  }

  // create device pointers
  Samples *d_samples[NR_BUFFERS];

  // the planar data needs to be __half so if NR_BITS == 8 then
  // we need to convert
#if NR_BITS == 8
  __half *d_samples_converted[NR_BUFFERS];
  float *d_visibilities_converted[NR_BUFFERS];
#endif
  __half *d_samples_planar[NR_BUFFERS], *d_samples_planar_col_maj[NR_BUFFERS];
  Visibilities *d_visibilities[NR_BUFFERS];
  std::complex<float> *d_visibilities_permuted[NR_BUFFERS];
  __half *d_weights[NR_BUFFERS], *d_weights_updated[NR_BUFFERS],
      *d_weights_permuted[NR_BUFFERS];
  float *d_eigenvalues[NR_BUFFERS];
  BeamformedData *d_beamformed_data[NR_BUFFERS],
      *d_beamformed_data_output[NR_BUFFERS];

  size_t size_d_samples_planar, size_d_visibilities_permuted;
  // start with these events in done state.
  for (auto i = 0; i < NR_BUFFERS; ++i) {
    cudaMalloc((void **)&d_samples[i], sizeof(Samples));
#if NR_BITS == 8
    size_d_samples_planar = sizeof(Samples) * sizeof(__half) / sizeof(int8_t);
    size_d_visibilities_permuted =
        sizeof(Visibilities) * sizeof(float) / sizeof(int32_t);
#else

    size_d_samples_planar = sizeof(Samples);
    size_d_visibilities_permuted = sizeof(Visibilities);
#endif
    cudaMalloc((void **)&d_samples_planar[i], size_d_samples_planar);
    cudaMalloc((void **)&d_samples_planar_col_maj[i], size_d_samples_planar);
    cudaMalloc((void **)&d_visibilities[i], sizeof(Visibilities));
    cudaMalloc((void **)&d_visibilities_permuted[i],
               size_d_visibilities_permuted);
    cudaMalloc((void **)&d_weights[i],
               num_weights * sizeof(std::complex<__half>));
    cudaMalloc((void **)&d_weights_updated[i],
               num_weights * sizeof(std::complex<__half>));
    cudaMalloc((void **)&d_weights_permuted[i],
               num_weights * sizeof(std::complex<__half>));
    cudaMalloc((void **)&d_eigenvalues[i], sizeof(float) * num_eigen);
    cudaMalloc((void **)&d_beamformed_data[i], sizeof(BeamformedData));
    cudaMalloc((void **)&d_beamformed_data_output[i], sizeof(BeamformedData));

#if NR_BITS == 8
    cudaMalloc((void **)&d_samples_converted[i], size_d_samples_planar);
    cudaMalloc((void **)&d_visibilities_converted[i],
               size_d_visibilities_permuted);
#endif

    // transfer weights
    cudaMemcpy(d_weights[i], h_weights,
               sizeof(std::complex<__half>) * num_weights, cudaMemcpyDefault);
    cudaEventRecord(input_transfer_done[i], streams[i]);
  }

#if DEBUG == 1
  // allocate debug buffers
  __half *h_weights_updated, *h_weights_permuted, *h_samples_planar,
      *h_weights_check, *h_samples_planar_col_maj;
  Samples *h_samples_check;
  cudaMallocHost(&h_weights_updated,
                 num_weights * sizeof(std::complex<__half>));
  cudaMallocHost(&h_weights_permuted,
                 num_weights * sizeof(std::complex<__half>));
  cudaMallocHost(&h_samples_planar, size_d_samples_planar);
  cudaMallocHost(&h_samples_planar_col_maj, size_d_samples_planar);
  cudaMallocHost(&h_weights_check, num_weights * sizeof(std::complex<__half>));
  cudaMallocHost(&h_samples_check, sizeof(Samples));
#endif

  const __half alpha = __float2half(1.0f);
  const float alpha_32 = 1.0f;
  // c = channel
  // b = block
  // r = receiver
  // p = polarization
  // t = time
  // z = complex
  // l = baseline
  // m = beam
  // s = time consolidated <block x time>
  std::vector<int> modePacket{'c', 'b', 'r', 'p', 't', 'z'};
  std::vector<int> modePlanar{'c', 'p', 'z', 'r', 'b', 't'};
  // We need the planar samples matrix to be in column-major memory layout
  // which is equivalent to transposing time and receiver structure here.
  // We also squash the b,t axes into s = block * time
  // CCGLIB requires that we have BLOCK x COMPLEX x COL x ROW structure.
  std::vector<int> modePlanarCons = {'c', 'p', 'z', 'r', 's'};
  std::vector<int> modePlanarColMajCons = {'c', 'p', 'z', 's', 'r'};
  std::vector<int> modeVisCorr{'c', 'l', 'p', 'q', 'z'};
  std::vector<int> modeVisDecomp{'c', 'p', 'q', 'l', 'z'};
  // Convert back to interleaved instead of planar output.
  // This is not strictly necessary to do in the pipeline.
  std::vector<int> modeBeamCCGLIB{'c', 'p', 'z', 'm', 's'};
  std::vector<int> modeBeamOutput{'c', 'p', 'm', 's', 'z'};
  std::vector<int> modeWeightsInput{'c', 'p', 'm', 'r', 'z'};
  std::vector<int> modeWeightsCCGLIB{'c', 'p', 'z', 'm', 'r'};

  std::unordered_map<int, int64_t> extent;
  extent['c'] = NR_CHANNELS;
  extent['b'] = NR_BLOCKS_FOR_CORRELATION;
  extent['r'] = NR_RECEIVERS;
  extent['p'] = NR_POLARIZATIONS;
  extent['q'] = NR_POLARIZATIONS; // 2nd polarizations for baselines
  extent['t'] = NR_TIMES_PER_BLOCK;
  extent['z'] = 2; // real, imaginary
  extent['l'] = NR_BASELINES;
  extent['m'] = NR_BEAMS;
  extent['s'] = NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK;

  CutensorSetup tensor_16(extent, CUTENSOR_R_16F, 128);
  CutensorSetup tensor_32(extent, CUTENSOR_R_32F, 128);

  tensor_16.addTensor(modePacket, "packet");
  tensor_16.addTensor(modePlanar, "planar");
  tensor_16.addTensor(modePlanarCons, "planarCons");
  tensor_16.addTensor(modePlanarColMajCons, "planarColMajCons");

  tensor_16.addTensor(modeWeightsInput, "weightsInput");
  tensor_16.addTensor(modeWeightsCCGLIB, "weightsCCGLIB");
  tensor_32.addTensor(modeVisCorr, "visCorr");
  tensor_32.addTensor(modeVisDecomp, "visDecomp");

  tensor_32.addTensor(modeBeamCCGLIB, "beamCCGLIB");
  tensor_32.addTensor(modeBeamOutput, "beamOutput");

  tensor_16.addPermutation("packet", "planar", CUTENSOR_COMPUTE_DESC_16F,
                           "packetToPlanar");
  tensor_16.addPermutation("planarCons", "planarColMajCons",
                           CUTENSOR_COMPUTE_DESC_16F, "consToColMajCons");
  tensor_16.addPermutation("weightsInput", "weightsCCGLIB",
                           CUTENSOR_COMPUTE_DESC_16F, "weightsInputToCCGLIB");
  tensor_32.addPermutation("visCorr", "visDecomp", CUTENSOR_COMPUTE_DESC_32F,
                           "visCorrToDecomp");
  tensor_32.addPermutation("beamCCGLIB", "beamOutput",
                           CUTENSOR_COMPUTE_DESC_32F, "beamCCGLIBToOutput");
  printf("Initializing correlator\n");
  printf("NR_RECEIVERS: %u\n", NR_RECEIVERS);
  printf("NR_CHANNELS: %u\n", NR_CHANNELS);
  printf("NR_SAMPLES_PER_CHANNEL: %u\n",
         NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK);
  printf("NR_POLARIZATIONS: %u\n", NR_POLARIZATIONS);
  printf("NR_RECEIVERS_PER_BLOCK: %u\n", NR_RECEIVERS_PER_BLOCK);
  tcc::Correlator correlator(cu::Device(0), inputFormat, NR_RECEIVERS,
                             NR_CHANNELS,
                             NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK,
                             NR_POLARIZATIONS, NR_RECEIVERS_PER_BLOCK);

  printf("NR_BLOCKS_FOR_CORRELATION: %u\n", NR_BLOCKS_FOR_CORRELATION);
  printf("NR_TIMES_PER_BLOCK: %u\n", NR_TIMES_PER_BLOCK);
  printf("NR_ACTUAL_RECEIVERS: %u\n", NR_ACTUAL_RECEIVERS);
  printf("NR_BITS: %u\n", NR_BITS);
  printf("Launching processing loop...\n");
  int current_buffer = 0;
  // std::atomic is overkill right now but if we end up using multi-threading at
  // some point this sidesteps a race condition.
  std::atomic<int> last_frame_processed = 0;
  bool processing = true;

  CUdevice cu_device;
  cuDeviceGet(&cu_device, 0);

  std::vector<std::unique_ptr<ccglib::mma::GEMM>> gemm_handles;

  for (auto i = 0; i < NR_BUFFERS; ++i) {
    gemm_handles.emplace_back(std::make_unique<ccglib::mma::GEMM>(
        NR_CHANNELS * NR_POLARIZATIONS, NR_BEAMS,
        NR_TIMES_PER_BLOCK * NR_BLOCKS_FOR_CORRELATION, NR_RECEIVERS, cu_device,
        streams[i], ccglib::ValueType::float16, ccglib::mma::basic));
  }
  // Ensure all copying is done before processing loop starts.
  // This may not be necessary.
  cudaDeviceSynchronize();
  // Main processing loop.
  while (processing) {
    if (cudaEventQuery(input_transfer_done[current_buffer]) == cudaSuccess) {
      printf("Beginning new processing loop....\n");
      int next_frame_to_capture = last_frame_processed.fetch_add(1);
      printf("Next frame to capture is %u for stream %u\n",
             next_frame_to_capture, current_buffer);
      // Use this for the lambda function to capture current value.
      if (next_frame_to_capture + 1 >= nr_aggregated_packets) {
        processing = false;
        printf(
            "Finishing processing loop as next frame is %u which +1 is greater "
            "than or equal to the number of aggregated_packets %u\n",
            next_frame_to_capture, nr_aggregated_packets);
      }

      cudaMemcpyAsync(
          d_samples[current_buffer], (void *)&h_samples[next_frame_to_capture],
          sizeof(Samples), cudaMemcpyDefault, streams[current_buffer]);
      // Now we can start preparing the next buffer for transport to the GPU.
      cudaEventRecord(input_transfer_done[current_buffer],
                      streams[current_buffer]);

      correlator.launchAsync((CUstream)streams[current_buffer],
                             (CUdeviceptr)d_visibilities[current_buffer],
                             (CUdeviceptr)d_samples[current_buffer]);

#if NR_BITS == 8
      convert_int8_to_half((int8_t *)d_samples[current_buffer],
                           d_samples_converted[current_buffer],
                           sizeof(Samples) / sizeof(int8_t),
                           streams[current_buffer]);
      convert_int_to_float((int *)d_visibilities[current_buffer],
                           d_visibilities_converted[current_buffer],
                           sizeof(Visibilities) / sizeof(int32_t),
                           streams[current_buffer]);

      tensor_16.runPermutation(
          "packetToPlanar", alpha, d_samples_converted[current_buffer],
          d_samples_planar[current_buffer], streams[current_buffer]);
      tensor_32.runPermutation(
          "visCorrToDecomp", alpha_32,
          (float *)d_visibilities_converted[current_buffer],
          (float *)d_visibilities_permuted[current_buffer],
          streams[current_buffer]);
#elif NR_BITS == 16
      tensor_16.runPermutation(
          "packetToPlanar", alpha, (__half *)d_samples[current_buffer],
          d_samples_planar[current_buffer], streams[current_buffer]);
      tensor_32.runPermutation("visCorrToDecomp", alpha_32,
                               (float *)d_visibilities[current_buffer],
                               (float *)d_visibilities_permuted[current_buffer],
                               streams[current_buffer]);
#endif

      tensor_16.runPermutation(
          "consToColMajCons", alpha, d_samples_planar[current_buffer],
          d_samples_planar_col_maj[current_buffer], streams[current_buffer]);
      checkCudaCall(
          cudaMemcpyAsync((void *)&h_visibilities_output[next_frame_to_capture],
                          d_visibilities[current_buffer], sizeof(Visibilities),
                          cudaMemcpyDefault, streams[current_buffer]));
      // need to think how multiple channels / polarizations works here - do we
      // need to do multiple decompositions? Probably yes. We'll also need to
      // convert the visibilities from int32 -> float

      // d_eigendecomposition(d_eigenvalues[current_buffer], NR_RECEIVERS,
      //                      NR_CHANNELS, NR_POLARIZATIONS,
      //                     d_visibilities_permuted[current_buffer],
      //                   streams[current_buffer]);

#if NR_BITS == 8
      update_weights(
          d_weights[current_buffer], d_weights_updated[current_buffer],
          NR_BEAMS, NR_RECEIVERS, NR_CHANNELS, NR_POLARIZATIONS,
          d_eigenvalues[current_buffer],
          d_visibilities_converted[current_buffer], streams[current_buffer]);

#else

      update_weights(
          d_weights[current_buffer], d_weights_updated[current_buffer],
          NR_BEAMS, NR_RECEIVERS, NR_CHANNELS, NR_POLARIZATIONS,
          d_eigenvalues[current_buffer],
          (float *)d_visibilities[current_buffer], streams[current_buffer]);
#endif

      tensor_16.runPermutation("weightsInputToCCGLIB", alpha,
                               (__half *)d_weights_updated[current_buffer],
                               d_weights_permuted[current_buffer],
                               streams[current_buffer]);
#if DEBUG == 1
      cudaMemcpyAsync(h_weights_updated, d_weights_updated[current_buffer],
                      sizeof(std::complex<__half>) * num_weights,
                      cudaMemcpyDefault, streams[current_buffer]);
      cudaMemcpyAsync(h_weights_permuted, d_weights_permuted[current_buffer],
                      sizeof(std::complex<__half>) * num_weights,
                      cudaMemcpyDefault, streams[current_buffer]);
      cudaMemcpyAsync(h_samples_planar, d_samples_planar[current_buffer],
                      size_d_samples_planar, cudaMemcpyDefault,
                      streams[current_buffer]);
      cudaMemcpyAsync(
          h_samples_planar_col_maj, d_samples_planar_col_maj[current_buffer],
          size_d_samples_planar, cudaMemcpyDefault, streams[current_buffer]);
      cudaMemcpyAsync(h_weights_check, d_weights[current_buffer],
                      sizeof(std::complex<__half>) * num_weights,
                      cudaMemcpyDefault, streams[current_buffer]);
      cudaMemcpyAsync(h_samples_check, d_samples[current_buffer],
                      sizeof(Samples), cudaMemcpyDefault,
                      streams[current_buffer]);
#endif

      (*gemm_handles[current_buffer])
          .Run((CUdeviceptr)d_weights_permuted[current_buffer],
               (CUdeviceptr)d_samples_planar_col_maj[current_buffer],
               (CUdeviceptr)d_beamformed_data[current_buffer]);

      tensor_32.runPermutation(
          "beamCCGLIBToOutput", alpha_32,
          (float *)d_beamformed_data[current_buffer],
          (float *)d_beamformed_data_output[current_buffer],
          streams[current_buffer]);

      cudaMemcpyAsync(&h_beam_output[next_frame_to_capture],
                      d_beamformed_data_output[current_buffer],
                      sizeof(BeamformedData), cudaMemcpyDefault,
                      streams[current_buffer]);
    }

    current_buffer = (current_buffer + 1) % NR_BUFFERS;
  }
  printf("Synchronizing...\n");
  cudaDeviceSynchronize();

#if DEBUG == 1
  printf("weights original...\n");
  for (auto i = 0; i < num_weights; ++i) {
    const std::complex<__half> val = h_weights[i];
    printf("%u: %f + %f j\n", i, __half2float(val.real()),
           __half2float(val.imag()));
  }

  printf("weights check...\n");
  for (auto i = 0; i < num_weights; ++i) {
    const std::complex<__half> val = std::complex<__half>(
        h_weights_check[2 * i], h_weights_check[2 * i + 1]);
    printf("%u: %f + %f j\n", i, __half2float(val.real()),
           __half2float(val.imag()));
  }
  printf("weights updated...\n");
  for (auto i = 0; i < num_weights; ++i) {
    const std::complex<__half> val = std::complex<__half>{
        h_weights_updated[2 * i], h_weights_updated[2 * i + 1]};
    printf("%u: %f + %f j\n", i, __half2float(val.real()),
           __half2float(val.imag()));
  }

  printf("weights permuted...\n");

  for (auto i = 0; i < num_weights * 2; ++i) {
    const __half weight = h_weights_permuted[i];
    printf("%u: %f\n", i, __half2float(weight));
  }

  printf("data from GPU...\n");
  // print_nonzero_samples(h_samples_check);
#endif

  for (auto i = 0; i < NR_BUFFERS; ++i) {
    cudaFree(d_samples[i]);
    cudaFree(d_samples_planar[i]);
    cudaFree(d_samples_planar_col_maj[i]);
    cudaFree(d_visibilities[i]);
    cudaFree(d_beamformed_data[i]);
    cudaFree(d_beamformed_data_output[i]);
    cudaFree(d_eigenvalues[i]);
    cudaFree(d_weights[i]);
    cudaFree(d_weights_updated[i]);
    cudaFree(d_weights_permuted[i]);
    cudaStreamDestroy(streams[i]);
    cudaEventDestroy(input_transfer_done[i]);
  }

#if DEBUG == 1
  cudaFreeHost(h_samples_check);
  cudaFreeHost(h_samples_planar);
  cudaFreeHost(h_samples_planar_col_maj);
  cudaFreeHost(h_weights_check);
  cudaFreeHost(h_weights_updated);
  cudaFreeHost(h_weights_permuted);
#endif
}

template void beamform(std::complex<__half> *h_samples,
                       std::complex<__half> *h_weights,
                       std::complex<float> *h_beam_output,
                       std::complex<float> *h_visibilities_output,
                       const int nr_aggregated_packets);

template void beamform(std::complex<int8_t> *h_samples,
                       std::complex<__half> *h_weights,
                       std::complex<float> *h_beam_output,
                       std::complex<float> *h_visibilities_output,
                       const int nr_aggregated_packets);
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
