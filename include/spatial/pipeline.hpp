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

template <typename T> struct BeamWeightsT {
  std::complex<__half> weights[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
                              [T::NR_RECEIVERS];
};

struct BufferReleaseContext {
  ProcessorStateBase *state;
  size_t buffer_index;
};

struct OutputTransferCompleteContext {
  std::shared_ptr<Output> output;
  size_t block_index;
};

// Static function to be called by cudaLaunchHostFunc
static void release_buffer_host_func(void *data) {
  auto *ctx = static_cast<BufferReleaseContext *>(data);
  LOG_INFO("Releasing buffer #{}", ctx->buffer_index);
  ctx->state->release_buffer(ctx->buffer_index);
  delete ctx;
}

static void output_transfer_complete_host_func(void *data) {
  auto *ctx = static_cast<OutputTransferCompleteContext *>(data);
  LOG_INFO("Marking beam data output transfer for block #{} complete",
           ctx->block_index);
  ctx->output->register_beam_data_transfer_complete(ctx->block_index);
  delete ctx;
}

static void output_visibilities_transfer_complete_host_func(void *data) {
  auto *ctx = static_cast<OutputTransferCompleteContext *>(data);
  LOG_INFO("Marking output transfer for block #{} complete", ctx->block_index);
  ctx->output->register_visibilities_transfer_complete(ctx->block_index);
  delete ctx;
}

template <typename T> class LambdaGPUPipeline : public GPUPipeline {

private:
  int num_buffers;
  std::vector<cudaStream_t> streams;

  // We are converting it to fp16 so this should not be changable anymore.
  static constexpr int NR_TIMES_PER_BLOCK = 128 / 16; // NR_BITS;

  static constexpr int NR_BLOCKS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET /
      NR_TIMES_PER_BLOCK;
  static constexpr int NR_BASELINES =
      T::NR_PADDED_RECEIVERS * (T::NR_PADDED_RECEIVERS + 1) / 2;
  static constexpr int NR_TIME_STEPS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET;
  static constexpr int COMPLEX = 2;
  static constexpr int NR_EIGENVALUES =
      T::NR_PADDED_RECEIVERS * T::NR_CHANNELS * T::NR_POLARIZATIONS;
  static constexpr int NR_CORRELATED_BLOCKS_TO_ACCUMULATE =
      T::NR_CORRELATED_BLOCKS_TO_ACCUMULATE;

  inline static const __half alpha = __float2half(1.0f);
  static constexpr float alpha_32 = 1.0f;

  using CorrelatorInput =
      __half[T::NR_CHANNELS][NR_BLOCKS_FOR_CORRELATION][T::NR_PADDED_RECEIVERS]
            [T::NR_POLARIZATIONS][NR_TIMES_PER_BLOCK][COMPLEX];

  using CorrelatorOutput =
      float[T::NR_CHANNELS][NR_BASELINES][T::NR_POLARIZATIONS]
           [T::NR_POLARIZATIONS][COMPLEX];

  using Visibilities =
      std::complex<float>[T::NR_CHANNELS][NR_BASELINES][T::NR_POLARIZATIONS]
                         [T::NR_POLARIZATIONS];

  using BeamformerInput =
      __half[T::NR_CHANNELS][NR_BLOCKS_FOR_CORRELATION][T::NR_PADDED_RECEIVERS]
            [T::NR_POLARIZATIONS][NR_TIMES_PER_BLOCK][COMPLEX];

  using BeamformerOutput =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
           [NR_TIME_STEPS_FOR_CORRELATION][COMPLEX];

  using BeamWeights = BeamWeightsT<T>;

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

      {'c', T::NR_CHANNELS},
      {'b', NR_BLOCKS_FOR_CORRELATION},
      {'r', T::NR_RECEIVERS},
      {'d', T::NR_PADDED_RECEIVERS},
      {'p', T::NR_POLARIZATIONS},
      {'q', T::NR_POLARIZATIONS}, // 2nd polarization for baselines
      {'t', NR_TIMES_PER_BLOCK},
      {'z', 2}, // real, imaginary
      {'l', NR_BASELINES},
      {'m', T::NR_BEAMS},
      {'s', NR_BLOCKS_FOR_CORRELATION *NR_TIMES_PER_BLOCK},

  };

  CutensorSetup tensor_16;
  CutensorSetup tensor_32;

  int current_buffer;
  std::atomic<int> last_frame_processed;
  std::atomic<int> num_integrated_units_processed;
  std::atomic<int> num_correlation_units_integrated;

  std::vector<std::unique_ptr<ccglib::mma::GEMM>> gemm_handles;

  tcc::Correlator correlator;

  std::vector<typename T::PacketSamplesType *> d_samples_entry,
      d_samples_scaled;
  std::vector<typename T::HalfPacketSamplesType *> d_samples_half,
      d_samples_padding;
  std::vector<typename T::PaddedPacketSamplesType *> d_samples_padded,
      d_samples_reord; // This is not the right type for reord
                       // - but it will do I guess. Size will be correct.
  std::vector<CorrelatorInput *> d_correlator_input;
  std::vector<CorrelatorOutput *> d_correlator_output;

  std::vector<Visibilities *> d_visibilities_converted,
      d_visibilities_accumulator, d_visibilities_permuted;
  std::vector<BeamformerInput *> d_beamformer_input;
  std::vector<BeamformerOutput *> d_beamformer_output, d_beamformer_data_output;
  std::vector<__half *> d_samples_consolidated, d_samples_consolidated_col_maj,
      d_weights, d_weights_updated, d_weights_permuted;
  std::vector<typename T::PacketScalesType *> d_scales;
  std::vector<float *> d_eigenvalues;

  BeamWeights *h_weights;

  int visibilities_start_seq_num;
  int visibilities_end_seq_num;

public:
  void execute_pipeline(FinalPacketData *packet_data) override {

    if (state_ == nullptr) {
      std::logic_error("State has not been set on GPUPipeline object!");
    }
    const size_t start_seq_num = packet_data->start_seq_id;
    const size_t end_seq_num = packet_data->end_seq_id;
    if (visibilities_start_seq_num == -1) {
      visibilities_start_seq_num = packet_data->start_seq_id;
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
        typename T::PacketSamplesPlanarType, typename T::PacketScalesType,
        typename T::HalfPacketSamplesPlanarType, T::NR_CHANNELS,
        T::NR_POLARIZATIONS, T::NR_RECEIVERS, T::NR_TIME_STEPS_PER_PACKET,
        T::NR_PACKETS_FOR_CORRELATION>(
        (typename T::PacketSamplesPlanarType *)d_samples_entry[current_buffer],
        d_scales[current_buffer],
        (typename T::HalfPacketSamplesPlanarType *)
            d_samples_half[current_buffer],
        streams[current_buffer]);

    //  Reorder so that receiver is the slowest changing index so that we can
    //  pad it out.
    tensor_16.runPermutation(
        "packetToPadding", alpha, (__half *)d_samples_half[current_buffer],
        (__half *)d_samples_padding[current_buffer], streams[current_buffer]);

    // tensor copy into correct place in d_samples_padded
    cudaMemcpyAsync(d_samples_padded[current_buffer],
                    d_samples_padding[current_buffer],
                    sizeof(typename T::HalfPacketSamplesType),
                    cudaMemcpyDefault, streams[current_buffer]);
    cudaMemsetAsync(
        // need to convert to char which gives in terms of bytes.
        reinterpret_cast<char *>(d_samples_padded[current_buffer]) +
            sizeof(typename T::HalfPacketSamplesType),
        0,
        sizeof(typename T::PaddedPacketSamplesType) -
            sizeof(typename T::HalfPacketSamplesType),
        streams[current_buffer]);

    tensor_16.runPermutation(
        "paddedToCorrInput", alpha, (__half *)d_samples_padded[current_buffer],
        (__half *)d_correlator_input[current_buffer], streams[current_buffer]);

    correlator.launchAsync((CUstream)streams[current_buffer],
                           (CUdeviceptr)d_correlator_output[current_buffer],
                           (CUdeviceptr)d_correlator_input[current_buffer]);

    tensor_32.runPermutation("visCorrToDecomp", alpha_32,
                             (float *)d_correlator_output[current_buffer],
                             (float *)d_visibilities_permuted[current_buffer],
                             streams[current_buffer]);
    accumulate_visibilities((float *)d_correlator_output[current_buffer],
                            (float *)d_visibilities_accumulator[current_buffer],
                            2 * NR_BASELINES, streams[current_buffer]);
    int current_num_correlation_units_integrated =
        num_correlation_units_integrated.fetch_add(1);

    // Each buffer has its own visibilities and we need to combine them once
    // it's done.

    // Dump out integrated values to host
    // Perhaps assume this is a multiple of the number of packets
    // in the data to allow this to be checked only at the end.
    if (current_num_correlation_units_integrated >=
        NR_CORRELATED_BLOCKS_TO_ACCUMULATE - 1) {
      dump_visibilities(end_seq_num);
    }
    // These two can be combined.
    tensor_16.runPermutation("packetToPlanar", alpha,
                             (__half *)d_samples_half[current_buffer],
                             (__half *)d_samples_consolidated[current_buffer],
                             streams[current_buffer]);
    tensor_16.runPermutation(
        "consToColMajCons", alpha,
        (__half *)d_samples_consolidated[current_buffer],
        (__half *)d_samples_consolidated_col_maj[current_buffer],
        streams[current_buffer]);

    update_weights(d_weights[current_buffer], d_weights_updated[current_buffer],
                   T::NR_BEAMS, T::NR_RECEIVERS, T::NR_CHANNELS,
                   T::NR_POLARIZATIONS, d_eigenvalues[current_buffer],
                   (float *)d_visibilities_converted[current_buffer],
                   streams[current_buffer]);
    // this seems suboptimal - figure this out later.
    tensor_16.runPermutation("weightsInputToCCGLIB", alpha,
                             (__half *)d_weights_updated[current_buffer],
                             (__half *)d_weights_permuted[current_buffer],
                             streams[current_buffer]);

    (*gemm_handles[current_buffer])
        .Run((CUdeviceptr)d_weights_permuted[current_buffer],
             (CUdeviceptr)d_samples_consolidated_col_maj[current_buffer],
             (CUdeviceptr)d_beamformer_output[current_buffer]);

    tensor_32.runPermutation("beamCCGLIBToOutput", alpha_32,
                             (float *)d_beamformer_output[current_buffer],
                             (float *)d_beamformer_data_output[current_buffer],
                             streams[current_buffer]);

    if (output_ == nullptr) {
      LOG_WARN("No output is defined!");
    } else {
      size_t block_num =
          output_->register_beam_data_block(start_seq_num, end_seq_num);
      void *landing_pointer = output_->get_beam_data_landing_pointer(block_num);

      cudaMemcpyAsync(landing_pointer, d_beamformer_data_output[current_buffer],
                      sizeof(BeamformerOutput), cudaMemcpyDefault,
                      streams[current_buffer]);
      auto *output_ctx = new OutputTransferCompleteContext{
          .output = this->output_, .block_index = block_num};

      cudaLaunchHostFunc(streams[current_buffer],
                         output_transfer_complete_host_func, output_ctx);
      // Move arrivals to the output
      bool *arrivals_output_pointer =
          (bool *)output_->get_arrivals_data_landing_pointer(block_num);

      std::memcpy(arrivals_output_pointer, packet_data->get_arrivals_ptr(),
                  packet_data->get_arrivals_size());
      output_->register_arrivals_transfer_complete(block_num);
    }

    // debug_kernel_launch<T>(
    //     (typename T::PacketSamplesPlanarType
    //     *)d_samples_entry[current_buffer], (typename T::PacketScalesType
    //     *)d_scales[current_buffer], (typename T::HalfPacketSamplesPlanarType
    //     *)
    //         d_samples_half[current_buffer],
    //     (typename T::HalfPacketSamplesPlanarType *)
    //         d_samples_padding[current_buffer],

    //    (typename T::PaddedPacketSamplesPlanarType *)
    //        d_samples_padded[current_buffer],
    //    streams[current_buffer]);

    current_buffer = (current_buffer + 1) % num_buffers;
  };
  LambdaGPUPipeline(const int num_buffers, BeamWeightsT<T> *h_weights)

      : num_buffers(num_buffers), h_weights(h_weights),

        correlator(cu::Device(0), tcc::Format::fp16, T::NR_PADDED_RECEIVERS,
                   T::NR_CHANNELS,
                   NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK,
                   T::NR_POLARIZATIONS, T::NR_PADDED_RECEIVERS_PER_BLOCK),

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
                            sizeof(typename T::PacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_scaled[i],
                            sizeof(typename T::PacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_half[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_consolidated[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_consolidated_col_maj[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_padded[i],
                            sizeof(typename T::PaddedPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_padding[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_reord[i],
                            sizeof(typename T::PaddedPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_scales[i],
                            sizeof(typename T::PacketScalesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_weights[i], sizeof(BeamWeights)));
      CUDA_CHECK(
          cudaMalloc((void **)&d_weights_permuted[i], sizeof(BeamWeights)));
      CUDA_CHECK(
          cudaMalloc((void **)&d_weights_updated[i], sizeof(BeamWeights)));
      CUDA_CHECK(
          cudaMalloc((void **)&d_correlator_input[i], sizeof(CorrelatorInput)));
      CUDA_CHECK(cudaMalloc((void **)&d_correlator_output[i],
                            sizeof(CorrelatorOutput)));
      CUDA_CHECK(
          cudaMalloc((void **)&d_beamformer_input[i], sizeof(BeamformerInput)));
      CUDA_CHECK(cudaMalloc((void **)&d_beamformer_output[i],
                            sizeof(BeamformerOutput)));
      CUDA_CHECK(cudaMalloc((void **)&d_beamformer_data_output[i],
                            sizeof(BeamformerOutput)));
      CUDA_CHECK(cudaMalloc((void **)&d_visibilities_converted[i],
                            sizeof(Visibilities)));
      CUDA_CHECK(cudaMalloc((void **)&d_visibilities_accumulator[i],
                            sizeof(Visibilities)));
      CUDA_CHECK(cudaMalloc((void **)&d_visibilities_permuted[i],
                            sizeof(Visibilities)));
      CUDA_CHECK(cudaMalloc((void **)&d_eigenvalues[i],
                            sizeof(float) * T::NR_PADDED_RECEIVERS));
    }

    last_frame_processed = 0;
    num_integrated_units_processed = 0;
    num_correlation_units_integrated = 0;
    visibilities_start_seq_num = -1;
    visibilities_end_seq_num = -1;
    current_buffer = 0;
    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);

    //    tcc::Format inputFormat = tcc::Format::fp16;
    for (auto i = 0; i < num_buffers; ++i) {
      gemm_handles.emplace_back(std::make_unique<ccglib::mma::GEMM>(
          T::NR_CHANNELS * T::NR_POLARIZATIONS, T::NR_BEAMS,
          NR_TIMES_PER_BLOCK * NR_BLOCKS_FOR_CORRELATION, T::NR_RECEIVERS,
          cu_device, streams[i], ccglib::ValueType::float16,
          ccglib::mma::basic));
    }

    LOG_DEBUG("Copying weights...");
    for (auto i = 0; i < num_buffers; ++i) {
      cudaMemcpy(d_weights[i], h_weights, sizeof(BeamWeights),
                 cudaMemcpyDefault);
    }

    cudaDeviceSynchronize();
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
    // If there are visibilities in the accumulator on the GPU - dump them
    // out to disk. These will get tagged with a -1 end_seq_id currently
    // which is not fully ideal.
    if (visibilities_start_seq_num != -1) {
      dump_visibilities();
    }
    cudaDeviceSynchronize();

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

  void dump_visibilities(const int end_seq_num = -1) override {

    LOG_INFO("Dumping correlations to host...");
    int current_num_integrated_units_processed =
        num_integrated_units_processed.fetch_add(1);
    LOG_INFO("Current num integrated units processed is {}",
             current_num_integrated_units_processed);
    cudaDeviceSynchronize();
    for (auto i = 1; i < num_buffers; ++i) {
      accumulate_visibilities((float *)d_visibilities_accumulator[i],
                              (float *)d_visibilities_accumulator[0],
                              NR_BASELINES * 2, streams[current_buffer]);
    }
    cudaDeviceSynchronize();
    size_t block_num = output_->register_visibilities_block(
        visibilities_start_seq_num, end_seq_num);
    visibilities_start_seq_num = -1;
    void *landing_pointer =
        output_->get_visibilities_landing_pointer(block_num);
    cudaMemcpyAsync(landing_pointer, d_visibilities_accumulator[0],
                    sizeof(Visibilities), cudaMemcpyDefault, streams[0]);
    auto *output_ctx = new OutputTransferCompleteContext{
        .output = this->output_, .block_index = block_num};

    cudaLaunchHostFunc(streams[current_buffer],
                       output_visibilities_transfer_complete_host_func,
                       output_ctx);
    for (auto i = 0; i < num_buffers; ++i) {
      cudaMemsetAsync(d_visibilities_accumulator[i], 0, sizeof(Visibilities),
                      streams[i]);
    }
    num_correlation_units_integrated.store(0);
  };
};
