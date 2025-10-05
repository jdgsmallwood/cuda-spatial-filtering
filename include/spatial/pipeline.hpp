#pragma once
#include "spatial/packet_formats.hpp"
#include "spatial/spatial.hpp"
#include <complex>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

#include "ccglib/common/precision.h"
#include "spatial/logging.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline_base.hpp"
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
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <highfive/highfive.hpp>

template <int NR_LAMBDA_BITS, int NR_LAMBDA_TIME_STEPS_PER_PACKET,
          int NR_LAMBDA_PACKETS_FOR_CORRELATION, int NR_LAMBDA_RECEIVERS,
          int NR_PADDED_LAMBDA_RECEIVERS, int NR_LAMBDA_POLARIZATIONS,
          int NR_LAMBDA_BEAMS, int NR_LAMBDA_RECEIVERS_PER_BLOCK>
class LambdaGPUPipeline : public GPUPipeline {

private:
  int num_buffers;
  std::vector<cudaStream_t> streams;

  static constexpr int NR_LAMBDA_TIMES_PER_BLOCK = 128 / NR_LAMBDA_BITS;

  static constexpr int NR_LAMBDA_BLOCKS_FOR_CORRELATION =
      NR_LAMBDA_PACKETS_FOR_CORRELATION * NR_LAMBDA_TIME_STEPS_PER_PACKET /
      NR_LAMBDA_TIMES_PER_BLOCK;
  static constexpr int NR_LAMBDA_BASELINES =
      NR_PADDED_LAMBDA_RECEIVERS * (NR_PADDED_LAMBDA_RECEIVERS + 1) / 2;
  static constexpr int NR_LAMBDA_TIME_STEPS_FOR_CORRELATION =
      NR_LAMBDA_PACKETS_FOR_CORRELATION * NR_LAMBDA_TIME_STEPS_PER_PACKET;
  static constexpr int COMPLEX = 2;
  static constexpr int NR_EIGENVALUES =
      NR_PADDED_LAMBDA_RECEIVERS * NR_LAMBDA_CHANNELS * NR_LAMBDA_POLARIZATIONS;

  static const __half alpha;
  static constexpr float alpha_32 = 1.0f;

  using LambdaCorrelatorInput =
      std::complex<int8_t>[NR_LAMBDA_CHANNELS][NR_LAMBDA_BLOCKS_FOR_CORRELATION]
                          [NR_PADDED_LAMBDA_RECEIVERS][NR_LAMBDA_POLARIZATIONS]
                          [NR_LAMBDA_TIMES_PER_BLOCK];

  using LambdaCorrelatorOutput =
      std::complex<int32_t>[NR_LAMBDA_CHANNELS][NR_LAMBDA_BASELINES]
                           [NR_LAMBDA_POLARIZATIONS][NR_LAMBDA_POLARIZATIONS];

  using LambdaVisibilities =
      std::complex<float>[NR_LAMBDA_CHANNELS][NR_LAMBDA_BASELINES]
                         [NR_LAMBDA_POLARIZATIONS][NR_LAMBDA_POLARIZATIONS];

  using LambdaBeamformerInput =
      __half[NR_LAMBDA_CHANNELS][NR_LAMBDA_BLOCKS_FOR_CORRELATION]
            [NR_PADDED_LAMBDA_RECEIVERS][NR_LAMBDA_POLARIZATIONS]
            [NR_LAMBDA_TIMES_PER_BLOCK][COMPLEX];

  using LambdaBeamformerOutput =
      float[NR_LAMBDA_CHANNELS][NR_LAMBDA_POLARIZATIONS][NR_LAMBDA_BEAMS]
           [NR_LAMBDA_TIME_STEPS_FOR_CORRELATION];

  using LambdaBeamWeights =
      std::complex<__half>[NR_LAMBDA_CHANNELS][NR_LAMBDA_POLARIZATIONS]
                          [NR_LAMBDA_BEAMS][NR_LAMBDA_RECEIVERS];

  // c = channel
  // b = block
  // r = receiver
  // p = polarization
  // t = time
  // z = complex
  // l = baseline
  // m = beam
  // s = time consolidated <block x time>
  inline static const std::vector<int> modePacket{'c', 'b', 'r', 'p', 't', 'z'};
  inline static const std::vector<int> modePlanar{'c', 'p', 'z', 'r', 'b', 't'};
  // We need the planar samples matrix to be in column-major memory layout
  // which is equivalent to transposing time and receiver structure here.
  // We also squash the b,t axes into s = block * time
  // CCGLIB requires that we have BLOCK x COMPLEX x COL x ROW structure.
  inline static const std::vector<int> modePlanarCons = {'c', 'p', 'z', 'r',
                                                         's'};
  inline static const std::vector<int> modePlanarColMajCons = {'c', 'p', 'z',
                                                               's', 'r'};
  inline static const std::vector<int> modeVisCorr{'c', 'l', 'p', 'q', 'z'};
  inline static const std::vector<int> modeVisDecomp{'c', 'p', 'q', 'l', 'z'};
  // Convert back to interleaved instead of planar output.
  // This is not strictly necessary to do in the pipeline.
  inline static const std::vector<int> modeBeamCCGLIB{'c', 'p', 'z', 'm', 's'};
  inline static const std::vector<int> modeBeamOutput{'c', 'p', 'm', 's', 'z'};
  inline static const std::vector<int> modeWeightsInput{'c', 'p', 'm', 'r',
                                                        'z'};
  inline static const std::vector<int> modeWeightsCCGLIB{'c', 'p', 'z', 'm',
                                                         'r'};

  std::unordered_map<int, int64_t> extent;

  CutensorSetup tensor_16;
  CutensorSetup tensor_32;

  int current_buffer;
  std::atomic<int> last_frame_processed;
  std::atomic<int> num_integrated_units_processed;
  std::atomic<int> num_correlation_units_integrated;

  std::vector<std::unique_ptr<ccglib::mma::GEMM>> gemm_handles;

  tcc::Correlator correlator;

  std::vector<LambdaPacketSamples *> d_samples_entry, d_samples_scaled;
  std::vector<LambdaCorrelatorInput *> d_correlator_input;
  std::vector<LambdaCorrelatorOutput *> d_correlator_output;

  std::vector<LambdaVisibilities *> d_visibilities_converted,
      d_visibilities_accumulator;
  std::vector<LambdaBeamformerInput *> d_beamformer_input;
  std::vector<LambdaBeamformerOutput *> d_beamformer_output;
  std::vector<__half *> d_samples_converted, d_samples_converted_col_maj,
      d_weights, d_weights_updated, d_weights_permuted;
  std::vector<LambdaScales *> d_scales;
  std::vector<float *> d_eigenvalues;

  LambdaBeamWeights *h_weights;

public:
  void execute_pipeline(FinalPacketData *packet_data) override {

    LOG_INFO("Hello from LambdaGPUPipeline!");
    if (state_ == nullptr) {
      std::logic_error("State has not been set on GPUPipeline object!");
    }

    cudaMemcpyAsync(d_samples_entry[current_buffer],
                    (void *)packet_data->get_samples_ptr(),
                    sizeof(LambdaPacketSamples), cudaMemcpyDefault,
                    streams[current_buffer]);

    cudaLaunchHostFunc(
        streams[current_buffer, &packet_data],
        [this]() {
          LOG_INFO("Releasing buffer #{}", packet_data->buffer_index);
          state_->release_buffer(packet_data->buffer_index);
        },
        nullptr);

    correlator.launchAsync((CUstream)streams[current_buffer],
                           (CUdeviceptr)d_correlator_output[current_buffer],
                           (CUdeviceptr)d_samples_entry[current_buffer]);

    convert_int8_to_half(
        (int8_t *)d_samples_entry[current_buffer],
        d_samples_converted[current_buffer],
        /* number of samples (2x complex) */ sizeof(LambdaPacketSamples) /
            sizeof(int8_t),
        streams[current_buffer]);
    convert_int_to_float((int *)d_correlator_output[current_buffer],
                         d_visibilities_converted[current_buffer],
                         sizeof(Visibilities) / sizeof(int32_t),
                         streams[current_buffer]);

    tensor_16.runPermutation(
        "packetToPlanar", alpha, d_samples_converted[current_buffer],
        d_samples_scaled[current_buffer], streams[current_buffer]);
    tensor_32.runPermutation("visCorrToDecomp", alpha_32,
                             (float *)d_visibilities_converted[current_buffer],
                             (float *)d_visibilities_permuted[current_buffer],
                             streams[current_buffer]);
    accumulate_visibilities((float *)d_visibilities_converted[current_buffer],
                            (float *)d_visibilities_accumulator[current_buffer],
                            2 * spatial::NR_BASELINES, streams[current_buffer]);
    int current_num_correlation_units_integrated =
        num_correlation_units_integrated.fetch_add(1);

    // Each buffer has its own visibilities and we need to combine them once
    // it's done.

    // Dump out integrated values to host
    if (current_num_correlation_units_integrated >=
        NR_CORRELATION_BLOCKS_TO_INTEGRATE - 1) {
      LOG_INFO("Dumping correlations to host...");
      int current_num_integrated_units_processed =
          num_integrated_units_processed.fetch_add(1);
      LOG_INFO("Current num integrated units processed is {}",
               current_num_integrated_units_processed);
      cudaDeviceSynchronize();
      for (auto i = 1; i < NR_BUFFERS; ++i) {
        accumulate_visibilities((float *)d_visibilities_accumulator[i],
                                (float *)d_visibilities_accumulator[0],
                                spatial::NR_BASELINES * 2,
                                streams[current_buffer]);
      }
      checkCudaCall(cudaMemcpyAsync(
          h_visibilities_output[current_num_integrated_units_processed],
          d_visibilities_accumulator[0], size_d_visibilities_permuted,
          cudaMemcpyDefault, streams[current_buffer]));
      for (auto i = 0; i < NR_BUFFERS; ++i) {
        cudaMemsetAsync(d_visibilities_accumulator[i], 0,
                        size_d_visibilities_permuted, streams[i]);
      }
      cudaDeviceSynchronize();
      num_correlation_units_integrated.store(0);
    }

    tensor_16.runPermutation(
        "consToColMajCons", alpha, d_samples_scaled[current_buffer],
        d_samples_scaled_col_maj[current_buffer], streams[current_buffer]);

    update_weights(d_weights[current_buffer], d_weights_updated[current_buffer],
                   NR_LAMBDA_BEAMS, NR_LAMBDA_RECEIVERS, NR_LAMBDA_CHANNELS,
                   NR_LAMBDA_POLARIZATIONS, d_eigenvalues[current_buffer],
                   d_visibilities_converted[current_buffer],
                   streams[current_buffer]);
    // this seems suboptimal - figure this out later.
    tensor_16.runPermutation("weightsInputToCCGLIB", alpha,
                             (__half *)d_weights_updated[current_buffer],
                             d_weights_permuted[current_buffer],
                             streams[current_buffer]);

    (*gemm_handles[current_buffer])
        .Run((CUdeviceptr)d_weights_permuted[current_buffer],
             (CUdeviceptr)d_samples_scaled_col_maj[current_buffer],
             (CUdeviceptr)d_beamformer_output[current_buffer]);

    tensor_32.runPermutation("beamCCGLIBToOutput", alpha_32,
                             (float *)d_beamformer_output[current_buffer],
                             (float *)d_beamformed_data_output[current_buffer],
                             streams[current_buffer]);

    cudaMemcpyAsync(h_beam_output[next_frame_to_capture],
                    d_beamformed_data_output[current_buffer],
                    sizeof(BeamformedData), cudaMemcpyDefault,
                    streams[current_buffer]);

    current_buffer = (current_buffer + 1) % num_buffers;
  };
};
LambdaGPUPipeline(const int num_buffers, const LambdaBeamWeights *h_weights)

    : num_buffers(num_buffers), h_weights(h_weights) {

  alpha = __float2half(1.0f);

  streams.resize(num_buffers);
  d_weights.resize(num_buffers);
  d_weights_update.resize(num_buffers);
  d_weights_permuted.resize(num_buffers);
  d_samples_entry.resize(num_buffers);
  d_scales.resize(num_buffers);
  d_samples_scaled.resize(num_buffers);
  d_correlator_input.resize(num_buffers);
  d_correlator_output.resize(num_buffers);
  d_beamformer_input.resize(num_buffers);
  d_beamformer_output.resize(num_buffers);
  for (auto i = 0; i < num_buffers; ++i) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i]), cudaStreamNonBlocking);
    CUDA_CHECK(
        cudaMalloc((void **)&d_samples_entry[i], sizeof(LambdaPacketSamples)));
    CUDA_CHECK(
        cudaMalloc((void **)&d_samples_scaled[i], sizeof(LambdaPacketSamples)));
    CUDA_CHECK(cudaMalloc((void **)&d_scales[i], sizeof(LambdaScales)));
    CUDA_CHECK(cudaMalloc((void **)&d_weights[i], sizeof(BeamWeights)));
    CUDA_CHECK(
        cudaMalloc((void **)&d_weights_permuted[i], sizeof(BeamWeights)));
    CUDA_CHECK(cudaMalloc((void **)&d_weights_updated[i], sizeof(BeamWeights)));
    CUDA_CHECK(cudaMalloc((void **)&d_correlator_input[i],
                          sizeof(LambdaCorrelatorInput)));
    CUDA_CHECK(cudaMalloc((void **)&d_correlator_output[i],
                          sizeof(LambdaCorrelatorOutput)));
    CUDA_CHECK(cudaMalloc((void **)&d_beamformer_input[i],
                          sizeof(LambdaBeamformerInput)));
    CUDA_CHECK(cudaMalloc((void **)&d_beamformer_output[i],
                          sizeof(LambdaBeamformerOutput)));
  }

  extent['c'] = NR_LAMBDA_CHANNELS;
  extent['b'] = NR_LAMBDA_BLOCKS_FOR_CORRELATION;
  extent['r'] = NR_PADDED_LAMBDA_RECEIVERS;
  extent['p'] = NR_LAMBDA_POLARIZATIONS;
  extent['q'] = NR_LAMBDA_POLARIZATIONS; // 2nd
                                         // polarizations
                                         // for
                                         // baselines
  extent['t'] = NR_LAMBDA_TIMES_PER_BLOCK;
  extent['z'] = 2; // real, imaginary
  extent['l'] = NR_LAMBDA_BASELINES;
  extent['m'] = NR_LAMBDA_BEAMS;
  extent['s'] = NR_LAMBDA_BLOCKS_FOR_CORRELATION * NR_LAMBDA_TIMES_PER_BLOCK;

  tensor_16 = CutensorSetup(extent, CUTENSOR_R_16F, 128);
  tensor_32 = CutensorSetup(extent, CUTENSOR_R_32F, 128);

  last_frame_processed = 0;
  num_integrated_units_processed = 0;
  num_correlation_units_integrated = 0;
  current_buffer = 0;
  CUdevice cu_device;
  cuDeviceGet(&cu_device, 0);

  correlator = tcc::Correlator(
      cu::Device(0), inputFormat, NR_PADDED_LAMBDA_RECEIVERS,
      NR_LAMBDA_CHANNELS,
      NR_LAMBDA_BLOCKS_FOR_CORRELATION * NR_LAMBDA_TIMES_PER_BLOCK,
      NR_LAMBDA_POLARIZATIONS, NR_LAMBDA_RECEIVERS_PER_BLOCK);
  for (auto i = 0; i < num_buffers; ++i) {
    gemm_handles.emplace_back(std::make_unique<ccglib::mma::GEMM>(
        NR_LAMBDA_CHANNELS * NR_LAMBDA_POLARIZATIONS, NR_LAMBDA_BEAMS,
        NR_LAMBDA_TIMES_PER_BLOCK * NR_LAMBDA_BLOCKS_FOR_CORRELATION,
        NR_LAMBDA_RECEIVERS, cu_device, streams[i], ccglib::ValueType::float16,
        ccglib::mma::basic));
  }
  cudaDeviceSynchronize();

  LOG_DEBUG("Copying weights...");
  for (auto i = 0; i < num_buffers; ++i) {

    cudaMemcpy(d_weights[i], h_weights,
               sizeof(LambdaBeamWeights, cudaMemcpyDefault));
  }
};
~LambdaGPUPipeline() {

  for (auto stream : streams) {
    cudaStreamDestroy(stream);
  }

  for (auto sample : d_samples_entry) {
    cudaFree(sample);
  }

  for (auto scale : d_scales) {
    cudaFree(scale);
  }

  for (auto weight : d_weights) {
    cudaFree(weight);
  }

  for (auto weight : d_weights_updated) {
    cudaFree(weight);
  }

  for (auto weight : d_weights_permuted) {
    cudaFree(weight);
  }

  for (auto correlator_input : d_correlator_input) {
    cudaFree(correlator_input);
  }

  for (auto correlator_output : d_correlator_output) {
    cudaFree(correlator_output);
  }

  for (auto beamformer_input : d_beamformer_input) {
    cudaFree(beamformer_input);
  }
  for (auto beamformer_output : d_beamformer_output) {
    cudaFree(beamformer_output);
  }
};
}
;
