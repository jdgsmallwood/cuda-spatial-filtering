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

template <int NR_CHANNELS, int NR_RECEIVERS, int NR_POLARIZATIONS, int NR_BEAMS>
struct BeamWeights {

  std::complex<__half> weights[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS]
                              [NR_RECEIVERS];
};

struct BufferReleaseContext {
  ProcessorStateBase *state;
  size_t buffer_index;
};

// Static function to be called by cudaLaunchHostFunc
static void release_buffer_host_func(void *data) {
  auto *ctx = static_cast<BufferReleaseContext *>(data);
  LOG_INFO("Releasing buffer #{}", ctx->buffer_index);
  ctx->state->release_buffer(ctx->buffer_index);
  delete ctx;
}

template <int NR_LAMBDA_BITS, int NR_LAMBDA_CHANNELS,
          int NR_LAMBDA_TIME_STEPS_PER_PACKET,
          int NR_LAMBDA_PACKETS_FOR_CORRELATION, int NR_LAMBDA_RECEIVERS,
          int NR_PADDED_LAMBDA_RECEIVERS, int NR_LAMBDA_POLARIZATIONS,
          int NR_LAMBDA_BEAMS, int NR_LAMBDA_RECEIVERS_PER_BLOCK>
class LambdaGPUPipeline : public GPUPipeline {

private:
  int num_buffers;
  std::vector<cudaStream_t> streams;

  // We are converting it to fp16 so this should not be changable anymore.
  static constexpr int NR_LAMBDA_TIMES_PER_BLOCK = 128 / 16; // NR_LAMBDA_BITS;

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

  inline static const __half alpha = __float2half(1.0f);
  static constexpr float alpha_32 = 1.0f;

  using LambdaCorrelatorInput =
      __half[NR_LAMBDA_CHANNELS][NR_LAMBDA_BLOCKS_FOR_CORRELATION]
            [NR_PADDED_LAMBDA_RECEIVERS][NR_LAMBDA_POLARIZATIONS]
            [NR_LAMBDA_TIMES_PER_BLOCK][COMPLEX];

  using LambdaCorrelatorOutput =
      float[NR_LAMBDA_CHANNELS][NR_LAMBDA_BASELINES][NR_LAMBDA_POLARIZATIONS]
           [NR_LAMBDA_POLARIZATIONS][COMPLEX];

  using LambdaVisibilities =
      std::complex<float>[NR_LAMBDA_CHANNELS][NR_LAMBDA_BASELINES]
                         [NR_LAMBDA_POLARIZATIONS][NR_LAMBDA_POLARIZATIONS];

  using LambdaBeamformerInput =
      __half[NR_LAMBDA_CHANNELS][NR_LAMBDA_BLOCKS_FOR_CORRELATION]
            [NR_PADDED_LAMBDA_RECEIVERS][NR_LAMBDA_POLARIZATIONS]
            [NR_LAMBDA_TIMES_PER_BLOCK][COMPLEX];

  using LambdaBeamformerOutput =
      float[NR_LAMBDA_CHANNELS][NR_LAMBDA_POLARIZATIONS][NR_LAMBDA_BEAMS]
           [NR_LAMBDA_TIME_STEPS_FOR_CORRELATION][COMPLEX];

  using LambdaBeamWeights =
      BeamWeights<NR_LAMBDA_CHANNELS, NR_LAMBDA_RECEIVERS,
                  NR_LAMBDA_POLARIZATIONS, NR_LAMBDA_BEAMS>;

  // c = channel
  // b = block
  // r = receiver
  // d = padded receivers
  // p = polarization
  // q = second polarization
  // t = time
  // z = complex
  // l = baseline
  // m = beam
  // s = time consolidated <block x time>
  inline static const std::vector<int> modePacket{'c', 'b', 't', 'r', 'p', 'z'};

  inline static const std::vector<int> modePacketPadding{'r', 'c', 'b',
                                                         't', 'p', 'z'};
  inline static const std::vector<int> modePacketPadded{'d', 'c', 'b',
                                                        't', 'p', 'z'};
  inline static const std::vector<int> modeCorrelatorInput{'c', 'b', 'd',
                                                           'p', 't', 'z'};
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

  inline static const std::unordered_map<int, int64_t> extent = {

      {'c', NR_LAMBDA_CHANNELS},
      {'b', NR_LAMBDA_BLOCKS_FOR_CORRELATION},
      {'r', NR_LAMBDA_RECEIVERS},
      {'d', NR_PADDED_LAMBDA_RECEIVERS},
      {'p', NR_LAMBDA_POLARIZATIONS},
      {'q', NR_LAMBDA_POLARIZATIONS}, // 2nd polarization for baselines
      {'t', NR_LAMBDA_TIMES_PER_BLOCK},
      {'z', 2}, // real, imaginary
      {'l', NR_LAMBDA_BASELINES},
      {'m', NR_LAMBDA_BEAMS},
      {'s', NR_LAMBDA_BLOCKS_FOR_CORRELATION *NR_LAMBDA_TIMES_PER_BLOCK},

  };

  CutensorSetup tensor_16;
  CutensorSetup tensor_32;

  int current_buffer;
  std::atomic<int> last_frame_processed;
  std::atomic<int> num_integrated_units_processed;
  std::atomic<int> num_correlation_units_integrated;

  std::vector<std::unique_ptr<ccglib::mma::GEMM>> gemm_handles;

  tcc::Correlator correlator;

  std::vector<LambdaPacketSamplesT<int8_t> *> d_samples_entry, d_samples_scaled;
  std::vector<LambdaPacketSamplesT<__half> *> d_samples_half, d_samples_padding;
  std::vector<LambdaPacketSamplesT<__half, NR_PADDED_LAMBDA_RECEIVERS> *>
      d_samples_padded,
      d_samples_reord; // This is not the right type for reord
                       // - but it will do I guess. Size will be correct.
  std::vector<LambdaCorrelatorInput *> d_correlator_input;
  std::vector<LambdaCorrelatorOutput *> d_correlator_output;

  std::vector<LambdaVisibilities *> d_visibilities_converted,
      d_visibilities_accumulator, d_visibilities_permuted;
  std::vector<LambdaBeamformerInput *> d_beamformer_input;
  std::vector<LambdaBeamformerOutput *> d_beamformer_output,
      d_beamformer_data_output;
  std::vector<__half *> d_samples_consolidated, d_samples_consolidated_col_maj,
      d_weights, d_weights_updated, d_weights_permuted;
  std::vector<LambdaScales *> d_scales;
  std::vector<float *> d_eigenvalues;

  LambdaBeamWeights *h_weights;
  size_t NR_CORRELATION_BLOCKS_TO_INTEGRATE;

public:
  void execute_pipeline(FinalPacketData *packet_data) override {

    if (state_ == nullptr) {
      std::logic_error("State has not been set on GPUPipeline object!");
    }

    cudaMemcpyAsync(d_samples_entry[current_buffer],
                    (void *)packet_data->get_samples_ptr(),
                    packet_data->get_samples_elements_size(), cudaMemcpyDefault,
                    streams[current_buffer]);
    cudaMemcpyAsync(d_scales[current_buffer],
                    (void *)packet_data->get_scales_ptr(),
                    packet_data->get_scales_element_size(), cudaMemcpyDefault,
                    streams[current_buffer]);

    auto *ctx = new BufferReleaseContext{
        .state = this->state_, .buffer_index = packet_data->buffer_index};

    cudaLaunchHostFunc(streams[current_buffer], release_buffer_host_func, ctx);

    scale_and_convert_to_half<
        LambdaPacketSamplesPlanarT<int8_t>, LambdaScales,
        LambdaPacketSamplesPlanarT<__half>, NR_LAMBDA_CHANNELS,
        NR_LAMBDA_POLARIZATIONS, NR_LAMBDA_RECEIVERS,
        NR_LAMBDA_TIME_STEPS_PER_PACKET, NR_LAMBDA_PACKETS_FOR_CORRELATION>(
        (LambdaPacketSamplesPlanarT<int8_t> *)d_samples_entry[current_buffer],
        d_scales[current_buffer],
        (LambdaPacketSamplesPlanarT<__half> *)d_samples_half[current_buffer],
        streams[current_buffer]);

    //  Reorder so that receiver is the slowest changing index so that we can
    //  pad it out.
    tensor_16.runPermutation(
        "packetToPadding", alpha, d_samples_half[current_buffer],
        d_samples_padding[current_buffer], streams[current_buffer]);

    // tensor copy into correct place in d_samples_padded
    cudaMemcpyAsync(d_samples_padded[current_buffer],
                    d_samples_padding[current_buffer],
                    sizeof(LambdaPacketSamplesT<__half>), cudaMemcpyDefault,
                    streams[current_buffer]);
    cudaMemsetAsync(
        // need to convert to char which gives in terms of bytes.
        reinterpret_cast<char *>(d_samples_padded[current_buffer]) +
            sizeof(LambdaPacketSamplesT<__half>),
        0,
        sizeof(LambdaPacketSamplesT<__half, NR_PADDED_LAMBDA_RECEIVERS>) -
            sizeof(LambdaPacketSamplesT<__half>),
        streams[current_buffer]);

    tensor_16.runPermutation(
        "paddedToCorrInput", alpha, d_samples_padded[current_buffer],
        d_correlator_input[current_buffer], streams[current_buffer]);

    correlator.launchAsync((CUstream)streams[current_buffer],
                           (CUdeviceptr)d_correlator_output[current_buffer],
                           (CUdeviceptr)d_correlator_input[current_buffer]);

    tensor_32.runPermutation("visCorrToDecomp", alpha_32,
                             (float *)d_correlator_output[current_buffer],
                             (float *)d_visibilities_permuted[current_buffer],
                             streams[current_buffer]);
    accumulate_visibilities((float *)d_correlator_output[current_buffer],
                            (float *)d_visibilities_accumulator[current_buffer],
                            2 * NR_LAMBDA_BASELINES, streams[current_buffer]);
    int current_num_correlation_units_integrated =
        num_correlation_units_integrated.fetch_add(1);

    // Each buffer has its own visibilities and we need to combine them once
    // it's done.

    // Dump out integrated values to host
    // Perhaps assume this is a multiple of the number of packets
    // in the data to allow this to be checked only at the end.
    if (current_num_correlation_units_integrated >=
        NR_CORRELATION_BLOCKS_TO_INTEGRATE - 1) {
      LOG_INFO("Dumping correlations to host...");
      int current_num_integrated_units_processed =
          num_integrated_units_processed.fetch_add(1);
      LOG_INFO("Current num integrated units processed is {}",
               current_num_integrated_units_processed);
      cudaDeviceSynchronize();
      for (auto i = 1; i < num_buffers; ++i) {
        accumulate_visibilities((float *)d_visibilities_accumulator[i],
                                (float *)d_visibilities_accumulator[0],
                                NR_LAMBDA_BASELINES * 2,
                                streams[current_buffer]);
      }
      // checkCudaCall(cudaMemcpyAsync(
      //     h_visibilities_output[current_num_integrated_units_processed],
      //     d_visibilities_accumulator[0], sizeof(LambdaVisibilities),
      //     cudaMemcpyDefault, streams[current_buffer]));
      for (auto i = 0; i < num_buffers; ++i) {
        cudaMemsetAsync(d_visibilities_accumulator[i], 0,
                        sizeof(LambdaVisibilities), streams[i]);
      }
      cudaDeviceSynchronize();
      num_correlation_units_integrated.store(0);
    }

    tensor_16.runPermutation(
        "packetToPlanar", alpha, d_samples_half[current_buffer],
        d_samples_consolidated[current_buffer], streams[current_buffer]);
    tensor_16.runPermutation("consToColMajCons", alpha,
                             d_samples_consolidated[current_buffer],
                             d_samples_consolidated_col_maj[current_buffer],
                             streams[current_buffer]);

    update_weights(d_weights[current_buffer], d_weights_updated[current_buffer],
                   NR_LAMBDA_BEAMS, NR_LAMBDA_RECEIVERS, NR_LAMBDA_CHANNELS,
                   NR_LAMBDA_POLARIZATIONS, d_eigenvalues[current_buffer],
                   (float *)d_visibilities_converted[current_buffer],
                   streams[current_buffer]);
    // this seems suboptimal - figure this out later.
    tensor_16.runPermutation("weightsInputToCCGLIB", alpha,
                             (__half *)d_weights_updated[current_buffer],
                             d_weights_permuted[current_buffer],
                             streams[current_buffer]);

    (*gemm_handles[current_buffer])
        .Run((CUdeviceptr)d_weights_permuted[current_buffer],
             (CUdeviceptr)d_samples_consolidated_col_maj[current_buffer],
             (CUdeviceptr)d_beamformer_output[current_buffer]);

    tensor_32.runPermutation("beamCCGLIBToOutput", alpha_32,
                             (float *)d_beamformer_output[current_buffer],
                             (float *)d_beamformer_data_output[current_buffer],
                             streams[current_buffer]);

    // cudaMemcpyAsync(h_beam_output[next_frame_to_capture],
    //                 d_beamformer_data_output[current_buffer],
    //                 sizeof(BeamformedData), cudaMemcpyDefault,
    //                 streams[current_buffer]);

    current_buffer = (current_buffer + 1) % num_buffers;
  };
  LambdaGPUPipeline(
      const int num_buffers,
      BeamWeights<NR_LAMBDA_CHANNELS, NR_LAMBDA_RECEIVERS,
                  NR_LAMBDA_POLARIZATIONS, NR_LAMBDA_BEAMS> *h_weights,
      size_t nr_correlation_blocks_to_integrate)

      : num_buffers(num_buffers), h_weights(h_weights),
        NR_CORRELATION_BLOCKS_TO_INTEGRATE(nr_correlation_blocks_to_integrate),

        correlator(cu::Device(0), tcc::Format::fp16, NR_PADDED_LAMBDA_RECEIVERS,
                   NR_LAMBDA_CHANNELS,
                   NR_LAMBDA_BLOCKS_FOR_CORRELATION * NR_LAMBDA_TIMES_PER_BLOCK,
                   NR_LAMBDA_POLARIZATIONS, NR_LAMBDA_RECEIVERS_PER_BLOCK),

        tensor_16(extent, CUTENSOR_R_16F, 128),
        tensor_32(extent, CUTENSOR_R_32F, 128)

  {
    streams.resize(num_buffers);
    d_weights.resize(num_buffers);
    d_weights_updated.resize(num_buffers);
    d_weights_permuted.resize(num_buffers);
    d_samples_entry.resize(num_buffers);
    d_scales.resize(num_buffers);
    d_samples_scaled.resize(num_buffers);
    d_samples_half.resize(num_buffers);
    d_samples_padded.resize(num_buffers);
    d_samples_padding.resize(num_buffers);
    d_samples_consolidated.resize(num_buffers);
    d_samples_consolidated_col_maj.resize(num_buffers);
    d_samples_reord.resize(num_buffers);
    d_correlator_input.resize(num_buffers);
    d_correlator_output.resize(num_buffers);
    d_beamformer_input.resize(num_buffers);
    d_beamformer_output.resize(num_buffers);
    d_beamformer_data_output.resize(num_buffers);
    d_visibilities_accumulator.resize(num_buffers);
    d_visibilities_converted.resize(num_buffers);
    d_visibilities_permuted.resize(num_buffers);
    d_eigenvalues.resize(num_buffers);
    for (auto i = 0; i < num_buffers; ++i) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_entry[i],
                            sizeof(LambdaPacketSamplesT<int8_t>)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_scaled[i],
                            sizeof(LambdaPacketSamplesT<int8_t>)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_half[i],
                            sizeof(LambdaPacketSamplesT<__half>)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_consolidated[i],
                            sizeof(LambdaPacketSamplesT<__half>)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_consolidated_col_maj[i],
                            sizeof(LambdaPacketSamplesT<__half>)));
      CUDA_CHECK(cudaMalloc(
          (void **)&d_samples_padded[i],
          sizeof(LambdaPacketSamplesT<__half, NR_PADDED_LAMBDA_RECEIVERS>)));
      CUDA_CHECK(cudaMalloc(
          (void **)&d_samples_padding[i],
          sizeof(LambdaPacketSamplesT<__half, NR_LAMBDA_RECEIVERS>)));
      CUDA_CHECK(cudaMalloc(
          (void **)&d_samples_reord[i],
          sizeof(LambdaPacketSamplesT<__half, NR_PADDED_LAMBDA_RECEIVERS>)));
      CUDA_CHECK(cudaMalloc((void **)&d_scales[i], sizeof(LambdaScales)));
      CUDA_CHECK(cudaMalloc((void **)&d_weights[i], sizeof(LambdaBeamWeights)));
      CUDA_CHECK(cudaMalloc((void **)&d_weights_permuted[i],
                            sizeof(LambdaBeamWeights)));
      CUDA_CHECK(cudaMalloc((void **)&d_weights_updated[i],
                            sizeof(LambdaBeamWeights)));
      CUDA_CHECK(cudaMalloc((void **)&d_correlator_input[i],
                            sizeof(LambdaCorrelatorInput)));
      CUDA_CHECK(cudaMalloc((void **)&d_correlator_output[i],
                            sizeof(LambdaCorrelatorOutput)));
      CUDA_CHECK(cudaMalloc((void **)&d_beamformer_input[i],
                            sizeof(LambdaBeamformerInput)));
      CUDA_CHECK(cudaMalloc((void **)&d_beamformer_output[i],
                            sizeof(LambdaBeamformerOutput)));
      CUDA_CHECK(cudaMalloc((void **)&d_beamformer_data_output[i],
                            sizeof(LambdaBeamformerOutput)));
      CUDA_CHECK(cudaMalloc((void **)&d_visibilities_converted[i],
                            sizeof(LambdaVisibilities)));
      CUDA_CHECK(cudaMalloc((void **)&d_visibilities_accumulator[i],
                            sizeof(LambdaVisibilities)));
      CUDA_CHECK(cudaMalloc((void **)&d_visibilities_permuted[i],
                            sizeof(LambdaVisibilities)));
      CUDA_CHECK(cudaMalloc((void **)&d_eigenvalues[i],
                            sizeof(float) * NR_PADDED_LAMBDA_RECEIVERS));
    }

    last_frame_processed = 0;
    num_integrated_units_processed = 0;
    num_correlation_units_integrated = 0;
    current_buffer = 0;
    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);

    //    tcc::Format inputFormat = tcc::Format::fp16;
    for (auto i = 0; i < num_buffers; ++i) {
      gemm_handles.emplace_back(std::make_unique<ccglib::mma::GEMM>(
          NR_LAMBDA_CHANNELS * NR_LAMBDA_POLARIZATIONS, NR_LAMBDA_BEAMS,
          NR_LAMBDA_TIMES_PER_BLOCK * NR_LAMBDA_BLOCKS_FOR_CORRELATION,
          NR_LAMBDA_RECEIVERS, cu_device, streams[i],
          ccglib::ValueType::float16, ccglib::mma::basic));
    }
    cudaDeviceSynchronize();

    LOG_DEBUG("Copying weights...");
    for (auto i = 0; i < num_buffers; ++i) {
      cudaMemcpy(d_weights[i], h_weights, sizeof(LambdaBeamWeights),
                 cudaMemcpyDefault);
    }

    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modePacketPadding, "packet_padding");
    tensor_16.addTensor(modePacketPadded, "packet_padded");
    tensor_16.addTensor(modeCorrelatorInput, "corr_input");
    tensor_16.addTensor(modePlanar, "planar");
    tensor_16.addTensor(modePlanarCons, "planarCons");
    tensor_16.addTensor(modePlanarColMajCons, "planarColMajCons");

    tensor_16.addTensor(modeWeightsInput, "weightsInput");
    tensor_16.addTensor(modeWeightsCCGLIB, "weightsCCGLIB");
    tensor_32.addTensor(modeVisCorr, "visCorr");
    tensor_32.addTensor(modeVisDecomp, "visDecomp");

    tensor_32.addTensor(modeBeamCCGLIB, "beamCCGLIB");
    tensor_32.addTensor(modeBeamOutput, "beamOutput");

    tensor_16.addPermutation("packet", "packet_padding",
                             CUTENSOR_COMPUTE_DESC_16F, "packetToPadding");

    tensor_16.addPermutation("packet_padded", "corr_input",
                             CUTENSOR_COMPUTE_DESC_16F, "paddedToCorrInput");
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

    for (auto samples_padding : d_samples_padding) {
      cudaFree(samples_padding);
    }

    for (auto samples_padded : d_samples_padded) {
      cudaFree(samples_padded);
    }

    for (auto samples_consolidated : d_samples_consolidated) {
      cudaFree(samples_consolidated);
    }

    for (auto samples_consolidated_col_maj : d_samples_consolidated_col_maj) {
      cudaFree(samples_consolidated_col_maj);
    }

    for (auto eigenvalues : d_eigenvalues) {
      cudaFree(eigenvalues);
    }
  };
};
