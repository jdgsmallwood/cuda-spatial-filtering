#pragma once
#include "spatial/deripple.hpp"
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <mutex>

// Compact the coarse-major retained fine-channel axis for PSRDADA. With the
// 32/27 PFB and 28 retained bins, adjacent coarse channels contain two copies
// of the same boundary-centre frequency. Keep all copies through calibration
// and beamforming, then omit the first bin of coarse channels 1..N-1 so DSPSR
// receives the 40*27+1 unique frequencies as one uniform axis.
template <size_t NR_INPUT_CHANNELS, size_t NR_OUTPUT_CHANNELS,
          size_t NR_EFFECTIVE_FINE_CHANNELS, size_t NR_TIMES,
          size_t NR_POLS, size_t NR_BEAMS>
__global__ void compact_shared_fine_boundaries_kernel(const float *input,
                                                       float *output) {
  constexpr size_t INNER = NR_POLS * 2;
  constexpr size_t N = NR_BEAMS * NR_TIMES * NR_OUTPUT_CHANNELS * INNER;
  const size_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= N)
    return;

  const size_t scalar_in_pol = output_index % INNER;
  const size_t output_channel = (output_index / INNER) % NR_OUTPUT_CHANNELS;
  const size_t beam_time = output_index / (INNER * NR_OUTPUT_CHANNELS);

  size_t input_channel = output_channel;
  if constexpr (NR_OUTPUT_CHANNELS != NR_INPUT_CHANNELS) {
    if (output_channel >= NR_EFFECTIVE_FINE_CHANNELS) {
      const size_t after_first_coarse =
          output_channel - NR_EFFECTIVE_FINE_CHANNELS;
      const size_t coarse =
          1 + after_first_coarse / (NR_EFFECTIVE_FINE_CHANNELS - 1);
      const size_t fine =
          1 + after_first_coarse % (NR_EFFECTIVE_FINE_CHANNELS - 1);
      input_channel = coarse * NR_EFFECTIVE_FINE_CHANNELS + fine;
    }
  }

  output[output_index] = input[(beam_time * NR_INPUT_CHANNELS + input_channel) *
                               INNER + scalar_in_pol];
}

template <size_t NR_INPUT_CHANNELS, size_t NR_OUTPUT_CHANNELS,
          size_t NR_EFFECTIVE_FINE_CHANNELS, size_t NR_TIMES,
          size_t NR_POLS, size_t NR_BEAMS>
__global__ void detect_intensity_and_compact_shared_fine_boundaries_kernel(
    const float *input, float *output) {
  constexpr size_t N = NR_BEAMS * NR_TIMES * NR_OUTPUT_CHANNELS;
  const size_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= N)
    return;

  const size_t output_channel = output_index % NR_OUTPUT_CHANNELS;
  const size_t beam_time = output_index / NR_OUTPUT_CHANNELS;
  size_t input_channel = output_channel;
  if constexpr (NR_OUTPUT_CHANNELS != NR_INPUT_CHANNELS) {
    if (output_channel >= NR_EFFECTIVE_FINE_CHANNELS) {
      const size_t after_first_coarse =
          output_channel - NR_EFFECTIVE_FINE_CHANNELS;
      const size_t coarse =
          1 + after_first_coarse / (NR_EFFECTIVE_FINE_CHANNELS - 1);
      const size_t fine =
          1 + after_first_coarse % (NR_EFFECTIVE_FINE_CHANNELS - 1);
      input_channel = coarse * NR_EFFECTIVE_FINE_CHANNELS + fine;
    }
  }

  const size_t input_base =
      (beam_time * NR_INPUT_CHANNELS + input_channel) * NR_POLS * 2;
  float intensity = 0.0f;
#pragma unroll
  for (size_t pol = 0; pol < NR_POLS; ++pol) {
    const float re = input[input_base + pol * 2];
    const float im = input[input_base + pol * 2 + 1];
    intensity += re * re + im * im;
  }
  output[output_index] = intensity;
}

template <typename T, bool RFI_MITIGATE = false,
          bool DETECTED_INTENSITY = false>
class LambdaPulsarFoldPipeline : public GPUPipeline {
  static_assert((T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET) %
                        T::NR_FINE_CHANNELS ==
                    0,
                "NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET must divide evenly "
                "by NR_FINE_CHANNELS");

private:
  static constexpr int NR_TIMES_PER_BLOCK = 128 / 16; // NR_BITS;

  static constexpr int NR_BLOCKS_FOR_CORRELATION =
      (T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET /
       T::NR_FINE_CHANNELS) /
      NR_TIMES_PER_BLOCK;
  static constexpr int NR_TIME_STEPS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET;
  static constexpr int NR_TIME_STEPS_PER_FINE_CHANNEL = T::NR_TIME_STEPS_PER_FINE_CHANNEL;
  static_assert(NR_TIME_STEPS_PER_FINE_CHANNEL == NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK);
  static constexpr int COMPLEX = 2;

  static constexpr int NR_BASELINES =
      T::NR_PADDED_RECEIVERS * (T::NR_PADDED_RECEIVERS + 1) / 2;
  static constexpr int NR_UNPADDED_BASELINES =
      T::NR_RECEIVERS * (T::NR_RECEIVERS + 1) / 2;
  // cuSOLVER batch size: one (NR_RECEIVERS × NR_RECEIVERS) matrix per
  // channel × pol × pol, matching LambdaGPUPipeline exactly.
  static constexpr int CUSOLVER_BATCH_SIZE =
      T::NR_CHANNELS * T::NR_POLARIZATIONS;

  // -------------------------------------------------------------------------
  // Array-type aliases
  // -------------------------------------------------------------------------
  using CorrelatorInput =
      __half[T::NR_CHANNELS][NR_BLOCKS_FOR_CORRELATION][T::NR_PADDED_RECEIVERS]
            [T::NR_POLARIZATIONS][NR_TIMES_PER_BLOCK][COMPLEX];

  using CorrelatorOutput =
      float[T::NR_CHANNELS][NR_BASELINES][T::NR_POLARIZATIONS]
           [T::NR_POLARIZATIONS][COMPLEX];

  using Visibilities =
      std::complex<float>[T::NR_CHANNELS][NR_BASELINES][T::NR_POLARIZATIONS]
                         [T::NR_POLARIZATIONS];

  using TrimmedVisibilities =
      std::complex<float>[T::NR_CHANNELS][NR_UNPADDED_BASELINES]
                         [T::NR_POLARIZATIONS];

  // Full NR_RECEIVERS × NR_RECEIVERS matrices (one per channel × pol × pol),
  // laid out as a flat batch for cuSOLVER / cuBLAS.
  // Shape: [CUSOLVER_BATCH_SIZE][NR_RECEIVERS][NR_RECEIVERS]
  using DecompositionVisibilities =
      std::complex<float>[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                         [T::NR_RECEIVERS];

  // Eigenvalues: one real vector of length NR_RECEIVERS per batch element.
  using Eigenvalues =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS];

  using ProjectionMatrix =
      std::complex<__half>[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                          [T::NR_RECEIVERS];
  using FloatProjectionMatrix =
      std::complex<float>[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                         [T::NR_RECEIVERS];

  struct RFIMitigatedT {
    static constexpr size_t NR_CHANNELS = T::NR_CHANNELS;
    static constexpr size_t NR_POLARIZATIONS = T::NR_POLARIZATIONS;
    static constexpr size_t NR_BEAMS = 2 * T::NR_BEAMS;
    static constexpr size_t NR_RECEIVERS = T::NR_RECEIVERS;
  };

  using RFIMitigatedBeamWeights = BeamWeightsT<RFIMitigatedT>;

  static constexpr int num_beams = T::NR_BEAMS * (RFI_MITIGATE ? 2 : 1);
  using BeamformerOutput = float[T::NR_CHANNELS][T::NR_POLARIZATIONS][num_beams]
                                [NR_TIME_STEPS_PER_FINE_CHANNEL][COMPLEX];
  using BeamOutput = float[num_beams][NR_TIME_STEPS_PER_FINE_CHANNEL]
                          [T::NR_CHANNELS][T::NR_POLARIZATIONS][COMPLEX];
  static constexpr size_t NR_DADA_CHANNELS =
      T::NR_FINE_CHANNELS > 1
          ? T::NR_FPGA_CHANNELS * (T::NR_EFFECTIVE_FINE_CHANNELS - 1) + 1
          : T::NR_CHANNELS;
  using DadaVoltageOutput =
      float[num_beams][NR_TIME_STEPS_PER_FINE_CHANNEL][NR_DADA_CHANNELS]
           [T::NR_POLARIZATIONS][COMPLEX];
  using DadaIntensityOutput =
      float[num_beams][NR_TIME_STEPS_PER_FINE_CHANNEL][NR_DADA_CHANNELS];
  using DadaBeamOutput = std::conditional_t<DETECTED_INTENSITY,
                                             DadaIntensityOutput,
                                             DadaVoltageOutput>;

  using BeamWeights = BeamWeightsT<T>;
  using ChosenBeamWeights =
      std::conditional_t<RFI_MITIGATE, RFIMitigatedBeamWeights, BeamWeights>;
  bool header_written;

  struct PipelineResources {
    cudaStream_t stream;
    cudaStream_t host_stream;

    DevicePtr<typename T::InputPacketSamplesType> samples_entry;
    DevicePtr<typename T::PacketScalesType> scales;
    DevicePtr<typename T::HalfPacketSamplesType> samples_half,
        samples_pre_align;
    DevicePtr<typename T::HalfPacketAlignedSamplesType> samples_aligned,
        samples_reordered,
        samples_consolidated, samples_consolidated_col_maj, samples_padding;
    DevicePtr<typename T::PaddedPacketSamplesType> samples_padded;
    DevicePtr<typename FineChannelizer<T>::FilterInputType> channelizer_input;
    DevicePtr<typename FineChannelizer<T>::FilterOutputType> channelizer_output;
    DevicePtr<BeamOutput> beam_output;
    DevicePtr<BeamWeights> weights, weights_permuted;
    DevicePtr<BeamWeights> weights_updated;
    DevicePtr<RFIMitigatedBeamWeights> weights_rfi_mitigated;
    DevicePtr<ChosenBeamWeights> weights_beamformer;
    DevicePtr<BeamformerOutput> beamformer_output;

    // Correlator I/O
    DevicePtr<CorrelatorInput> correlator_input;
    DevicePtr<CorrelatorOutput> correlator_output;

    // Intermediate visibility layout buffers
    DevicePtr<Visibilities> visibilities_baseline;
    DevicePtr<TrimmedVisibilities> visibilities_trimmed_baseline;
    DevicePtr<TrimmedVisibilities> visibilities_trimmed;

    // Per-block correlation matrix (input to cuSOLVER, overwritten in-place
    // with eigenvectors on output).
    DevicePtr<DecompositionVisibilities> decomp_visibilities;
    DevicePtr<ProjectionMatrix> projection_matrix;
    DevicePtr<FloatProjectionMatrix> float_projection_matrix;

    // Per-block eigenvalues (ascending order from cuSOLVER).
    DevicePtr<Eigenvalues> eigenvalues;

    // cuSOLVER handles / workspace
    cusolverDnHandle_t cusolver_handle = nullptr;
    cusolverDnParams_t cusolver_params = nullptr;
    DevicePtr<int> cusolver_info; // [CUSOLVER_BATCH_SIZE]
    DevicePtr<void> cusolver_work_device;
    void *cusolver_work_host = nullptr;
    size_t cusolver_work_device_size = 0;
    size_t cusolver_work_host_size = 0;
    std::unique_ptr<ccglib::pipeline::Pipeline> gemm_handle;
    std::unique_ptr<ccglib::pipeline::Pipeline> gemm_weight_projection_handle;

    cublasHandle_t cublas_handle = nullptr;

    PipelineResources(CUdevice cu_device)
        : samples_entry(make_device_ptr<typename T::InputPacketSamplesType>()),
          scales(make_device_ptr<typename T::PacketScalesType>()),
          samples_half(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_pre_align(
              make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_aligned(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_reordered(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_consolidated(
              make_device_ptr_if<typename T::HalfPacketAlignedSamplesType,
                                 T::NR_FINE_CHANNELS == 1>()),
          samples_consolidated_col_maj(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_padding(
              make_device_ptr_if<typename T::HalfPacketAlignedSamplesType,
                                 RFI_MITIGATE && T::NR_FINE_CHANNELS == 1>()),
          channelizer_input(
              make_device_ptr<typename FineChannelizer<T>::FilterInputType>()),
          channelizer_output(
              make_device_ptr<typename FineChannelizer<T>::FilterOutputType>()),
          beam_output(make_device_ptr<BeamOutput>()),
          weights(make_device_ptr<BeamWeights>()),
          weights_permuted(make_device_ptr<BeamWeights>()),
          weights_updated(make_device_ptr_if<BeamWeights, RFI_MITIGATE>()),
          weights_rfi_mitigated(
              make_device_ptr_if<RFIMitigatedBeamWeights, RFI_MITIGATE>()),
          weights_beamformer(make_device_ptr<ChosenBeamWeights>()),
          beamformer_output(make_device_ptr<BeamformerOutput>()),
          samples_padded(
              make_device_ptr_if<typename T::PaddedPacketSamplesType,
                                 RFI_MITIGATE && T::NR_FINE_CHANNELS == 1>()),
          correlator_input(make_device_ptr_if<CorrelatorInput, RFI_MITIGATE>()),
          correlator_output(make_device_ptr_if<CorrelatorOutput, RFI_MITIGATE>()),
          visibilities_baseline(make_device_ptr_if<Visibilities, RFI_MITIGATE>()),
          visibilities_trimmed_baseline(
              make_device_ptr_if<TrimmedVisibilities, RFI_MITIGATE>()),
          visibilities_trimmed(
              make_device_ptr_if<TrimmedVisibilities, RFI_MITIGATE>()),
          decomp_visibilities(
              make_device_ptr_if<DecompositionVisibilities, RFI_MITIGATE>()),
          projection_matrix(make_device_ptr_if<ProjectionMatrix, RFI_MITIGATE>()),
          float_projection_matrix(
              make_device_ptr_if<FloatProjectionMatrix, RFI_MITIGATE>()),
          eigenvalues(make_device_ptr_if<Eigenvalues, RFI_MITIGATE>()),
          cusolver_info(
              make_device_ptr_if<int, RFI_MITIGATE>(
                  CUSOLVER_BATCH_SIZE * sizeof(int))) {
      // Stream Creation
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUDA_CHECK(
          cudaStreamCreateWithFlags(&host_stream, cudaStreamNonBlocking));

      if constexpr (RFI_MITIGATE) {
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
        CUSOLVER_CHECK(cusolverDnCreateParams(&cusolver_params));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle, stream));

        CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
            cusolver_handle, cusolver_params, CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_UPPER, T::NR_RECEIVERS, CUDA_C_32F,
            reinterpret_cast<void *>(decomp_visibilities.get()), T::NR_RECEIVERS,
            CUDA_R_32F, reinterpret_cast<void *>(eigenvalues.get()), CUDA_C_32F,
            &cusolver_work_device_size, &cusolver_work_host_size,
            CUSOLVER_BATCH_SIZE));

        cusolver_work_device = make_device_ptr<void>(cusolver_work_device_size);
        cusolver_work_host = std::malloc(cusolver_work_host_size);
      }

      const std::complex<float> alpha_ccglib = {1, 0};
      const std::complex<float> beta_ccglib = {0, 0};
      // GEMM Initialization
      // if RFI mitigate - we will have double the number of beams.
      gemm_handle = std::make_unique<ccglib::pipeline::Pipeline>(
          T::NR_CHANNELS * T::NR_POLARIZATIONS, num_beams,
          NR_TIMES_PER_BLOCK * NR_BLOCKS_FOR_CORRELATION, T::NR_RECEIVERS,
          cu_device, stream, ccglib::complex_planar, ccglib::complex_planar,
          ccglib::mma::row_major, ccglib::mma::col_major,
          ccglib::mma::row_major, ccglib::ValueType::float16,
          ccglib::ValueType::float32, ccglib::mma::opt, alpha_ccglib,
          beta_ccglib);

      if constexpr (RFI_MITIGATE) {
        gemm_weight_projection_handle =
            std::make_unique<ccglib::pipeline::Pipeline>(
              T::NR_CHANNELS * T::NR_POLARIZATIONS, T::NR_BEAMS,
              T::NR_RECEIVERS, T::NR_RECEIVERS, cu_device, stream,
              ccglib::complex_interleaved, ccglib::complex_interleaved,
              ccglib::mma::row_major, ccglib::mma::col_major,
              ccglib::mma::row_major, ccglib::ValueType::float16,
              ccglib::ValueType::float16, ccglib::mma::opt, alpha_ccglib,
              beta_ccglib);
      }
      CUBLAS_CHECK(cublasCreate(&cublas_handle));
      CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    }

    ~PipelineResources() {
      cudaStreamDestroy(stream);
      cudaStreamDestroy(host_stream);
      if (cusolver_handle)
        cusolverDnDestroy(cusolver_handle);
      if (cusolver_params)
        cusolverDnDestroyParams(cusolver_params);
      if (cusolver_work_host)
        std::free(cusolver_work_host);
      if (cublas_handle)
        cublasDestroy(cublas_handle);
    }
    PipelineResources(PipelineResources &&other) noexcept
        : stream(other.stream), host_stream(other.host_stream),
          samples_entry(std::move(other.samples_entry)),
          scales(std::move(other.scales)),
          samples_half(std::move(other.samples_half)),
          samples_reordered(std::move(other.samples_reordered)),
          samples_consolidated(std::move(other.samples_consolidated)),
          samples_consolidated_col_maj(
              std::move(other.samples_consolidated_col_maj)),
          beam_output(std::move(other.beam_output)),
          weights(std::move(other.weights)),
          weights_permuted(std::move(other.weights_permuted)),
          beamformer_output(std::move(other.beamformer_output)),
          gemm_handle(std::move(other.gemm_handle)),
          samples_padding(std::move(other.samples_padding)),
          samples_padded(std::move(other.samples_padded)),
          channelizer_input(std::move(other.channelizer_input)),
          channelizer_output(std::move(other.channelizer_output)),
          correlator_input(std::move(other.correlator_input)),
          correlator_output(std::move(other.correlator_output)),
          float_projection_matrix(std::move(other.float_projection_matrix)),
          projection_matrix(std::move(other.projection_matrix)),
          visibilities_baseline(std::move(other.visibilities_baseline)),
          visibilities_trimmed_baseline(
              std::move(other.visibilities_trimmed_baseline)),
          visibilities_trimmed(std::move(other.visibilities_trimmed)),
          decomp_visibilities(std::move(other.decomp_visibilities)),
          eigenvalues(std::move(other.eigenvalues)),
          cusolver_handle(other.cusolver_handle),
          cusolver_params(other.cusolver_params),
          cusolver_work_device(std::move(other.cusolver_work_device)),
          cusolver_work_host(other.cusolver_work_host),
          cusolver_work_device_size(other.cusolver_work_device_size),
          cusolver_work_host_size(other.cusolver_work_host_size),
          cusolver_info(std::move(other.cusolver_info))

    {
      other.stream = nullptr;
      other.host_stream = nullptr;
      other.cusolver_handle = nullptr;
      other.cusolver_params = nullptr;
      other.cusolver_work_host = nullptr;
    }

    PipelineResources &operator=(PipelineResources &&other) noexcept {
      if (this != &other) {
        if (stream)
          cudaStreamDestroy(stream);
        if (host_stream)
          cudaStreamDestroy(host_stream);

        stream = other.stream;
        host_stream = other.host_stream;
        samples_entry = std::move(other.samples_entry);
        scales = std::move(other.scales);
        samples_half = std::move(other.samples_half);
        samples_consolidated = std::move(other.samples_consolidated);
        samples_consolidated_col_maj =
            std::move(other.samples_consolidated_col_maj);
        beam_output = std::move(other.beam_output);
        weights = std::move(other.weights);
        weights_permuted = std::move(other.weights_permuted);
        beamformer_output = std::move(other.beamformer_output);
        gemm_handle = std::move(other.gemm_handle);

        other.stream = nullptr;

        other.host_stream = nullptr;
      }
      return *this;
    }

    PipelineResources(const PipelineResources &) = delete;
    PipelineResources &operator=(const PipelineResources &) = delete;
  };

  int num_buffers{1};
  std::vector<PipelineResources> buffers;

  cusolverEigMode_t cusolver_jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t cusolver_uplo = CUBLAS_FILL_MODE_UPPER;
  tcc::Correlator correlator;
  // We are converting it to fp16 so this should not be changable anymore.

  inline static const __half alpha = __float2half(1.0f);

  std::unordered_map<int, int> NR_SIGNAL_EIGENVECTORS;
  int min_freq_channel;

  static constexpr float alpha_32 = 1.0f;
  // a = unpadded baselines
  // b = block
  // c = channel
  // d = padded receivers
  // f = fpga
  // l = baseline
  // m = beam
  // n = receivers per packet
  // o = packets for correlation
  // p = polarization
  // q = second polarization
  // r = receiver
  // s = time consolidated <block x time>
  // t = times per block
  // u = time steps per packet
  // z = complex

  inline static const std::vector<int> modePacket{'C', 'y', 'f', 'u',
                                                  'n', 'p', 'z'};
  inline static const std::vector<int> modePlanar{'c', 'p', 'z', 'f',
                                                  'n', 'o', 'u'};

  inline static const std::vector<int> modePacketPreAlign{'f', 'y', 'u', 'C',
                                                          'n', 'p', 'z'};
  inline static const std::vector<int> modePacketAligned{'f', 'o', 'u', 'c',
                                                         'n', 'p', 'z'};
  // We need the planar samples matrix to be in column-major memory layout
  // which is equivalent to transposing time and receiver structure here.
  // We also squash the b,t axes into s = block * time
  // CCGLIB requires that we have BLOCK x COMPLEX x COL x ROW structure.
  inline static const std::vector<int> modePlanarCons = {'c', 'p', 'z',
                                                         'f', 'n', 's'};
  inline static const std::vector<int> modePlanarColMajCons = {'c', 'p', 'z',
                                                               's', 'f', 'n'};

  inline static const std::vector<int> modeBeamCCGLIB{'c', 'p', 'z', 'e', 's'};
  inline static const std::vector<int> modeBeamOutput{'e', 's', 'c', 'p', 'z'};
  inline static const std::vector<int> modeWeightsInput{'c', 'p', 'm', 'r',
                                                        'z'};
  inline static const std::vector<int> modeWeightsBeamMajor{'m', 'c', 'p', 'r',
                                                            'z'};

  inline static const std::vector<int> modeWeights2xBeamMajor{'e', 'c', 'p',
                                                              'r', 'z'};
  inline static const std::vector<int> modeWeightsCCGLIB{'c', 'p', 'z', 'e',
                                                         'r'};

  inline static const std::vector<int> modePacketPadding{'f', 'n', 'c', 'o',
                                                         'u', 'p', 'z'};
  inline static const std::vector<int> modePacketPadded{'d', 'c', 'b',
                                                        't', 'p', 'z'};
  inline static const std::vector<int> modeCorrelatorInput{'c', 'b', 'd',
                                                           'p', 't', 'z'};
  inline static const std::vector<int> modeVisCorr{'c', 'l', 'p', 'q', 'z'};
  inline static const std::vector<int> modeVisCorrBaseline{'p', 'q', 'l', 'c',
                                                           'z'};
  inline static const std::vector<int> modeVisCorrBaselineTrimmed{'p', 'a', 'c',
                                                                  'z'};
  inline static const std::vector<int> modeVisDecomp{'c', 'p', 'a', 'z'};

  inline static const std::unordered_map<int, int64_t> extent = {
      {'a', NR_UNPADDED_BASELINES},
      {'b', NR_BLOCKS_FOR_CORRELATION},
      {'c', T::NR_CHANNELS},
      {'C', T::NR_FPGA_CHANNELS},
      {'d', T::NR_PADDED_RECEIVERS},
      {'e', num_beams}, // rfi mitigated beam + original beam
      {'f', T::NR_FPGA_SOURCES},
      {'l', NR_BASELINES},
      {'m', T::NR_BEAMS},
      {'n', T::NR_RECEIVERS_PER_PACKET},
      {'o', T::NR_PACKETS_FOR_CORRELATION},
      {'p', T::NR_POLARIZATIONS},
      {'q', T::NR_POLARIZATIONS}, // 2nd polarization for baselines
      {'r', T::NR_RECEIVERS},
      {'s', NR_BLOCKS_FOR_CORRELATION *NR_TIMES_PER_BLOCK},
      {'t', NR_TIMES_PER_BLOCK},
      {'u', T::NR_TIME_STEPS_PER_PACKET},
      {'y', T::NR_PACKETS_FOR_CORRELATION + 2},
      {'z', 2}, // real, imaginary
  };

  CutensorSetup tensor_16;
  CutensorSetup tensor_32;

  int current_buffer;
  std::atomic<int> last_frame_processed;

  multilog_t *log;
  key_t dada_key;
  dada_hdu_t *hdu;
  key_t rfi_dada_key;
  dada_hdu_t *rfi_hdu;

  char *obs_header;

  BeamWeights *h_weights;
  // See the comment on this member in LambdaGPUPipeline (pipeline.hpp above)
  // -- same helper, same inert-when-unsteered contract.
  BeamSteering<T> beam_steering_;
  int *d_subpacket_delays;
  typename T::AntennaGains *d_gains;
  int *d_stream_perm_recv = nullptr;
  int *d_stream_perm_pol  = nullptr;
  std::unique_ptr<FineChannelizer<T>> channelizer_;
  std::vector<DevicePtr<DadaBeamOutput>> dada_beam_outputs_;
  bool async_dada_writer = false;
  std::vector<cudaEvent_t> dada_compute_done;
  cudaStream_t dada_copy_stream = nullptr;
  std::vector<bool> gpu_buffer_free;
  std::deque<int> dada_ready_queue;
  std::mutex dada_writer_mutex;
  std::condition_variable dada_writer_cv;
  std::thread dada_writer_thread;
  bool dada_writer_stopping = false;
  std::exception_ptr dada_writer_error;
  std::string deripple_response_id_;
  // Release packet-assembly buffers by polling one H2D completion event per
  // host buffer. CUDA host callbacks can starve when the DADA sink blocks or
  // another CUDA client (DSPSR) shares the device; once callbacks stop, all
  // input buffers remain owned and capture stalls despite zero NIC drops.
  std::vector<cudaEvent_t> host_buffer_copy_done;
  std::unique_ptr<std::atomic<bool>[]> host_buffer_release_pending;
  size_t host_buffer_release_count = 0;
  std::atomic<bool> buffer_reclaimer_running{false};
  std::thread buffer_reclaimer_thread;

  void run_dada_writer() {
    try {
      while (true) {
        int index;
        {
          std::unique_lock<std::mutex> lock(dada_writer_mutex);
          dada_writer_cv.wait(lock, [this] {
            return dada_writer_stopping || !dada_ready_queue.empty();
          });
          if (dada_ready_queue.empty())
            return;
          index = dada_ready_queue.front();
          dada_ready_queue.pop_front();
        }
        CUDA_CHECK(cudaEventSynchronize(dada_compute_done[index]));
        uint64_t block_id = 0;
        char *block = ipcio_open_block_write(hdu->data_block, &block_id);
        if (!block)
          throw std::runtime_error("pulsar DADA writer could not open a data block");
        const uint64_t block_size =
            ipcbuf_get_bufsz((ipcbuf_t *)hdu->data_block);
        CUDA_CHECK(cudaMemcpyAsync(block, dada_beam_outputs_[index].get(),
                                   block_size, cudaMemcpyDefault,
                                   dada_copy_stream));
        CUDA_CHECK(cudaStreamSynchronize(dada_copy_stream));
        if (ipcio_close_block_write(hdu->data_block, block_size) < 0)
          throw std::runtime_error("pulsar DADA writer could not close a data block");
        {
          std::lock_guard<std::mutex> lock(dada_writer_mutex);
          gpu_buffer_free[index] = true;
        }
        dada_writer_cv.notify_all();
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(dada_writer_mutex);
        dada_writer_error = std::current_exception();
      }
      dada_writer_cv.notify_all();
    }
  }

public:
  void set_stream_permutation(const std::vector<int> &recv_perm,
                              const std::vector<int> &pol_perm) override {
    constexpr int NR_RECVS = T::NR_FPGA_SOURCES * T::NR_RECEIVERS_PER_PACKET;
    constexpr int NR_PERM  = NR_RECVS * (int)T::NR_POLARIZATIONS;
    if ((int)recv_perm.size() != NR_PERM || (int)pol_perm.size() != NR_PERM)
      throw std::runtime_error(
          "set_stream_permutation: size must equal NR_FPGA_SOURCES * "
          "NR_RECEIVERS_PER_PACKET * NR_POLARIZATIONS");
    CUDA_CHECK(cudaMemcpy(d_stream_perm_recv, recv_perm.data(),
                          sizeof(int) * NR_PERM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_stream_perm_pol, pol_perm.data(),
                          sizeof(int) * NR_PERM, cudaMemcpyHostToDevice));
  }

  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {

    const int gpu_buffer_index = current_buffer;
    if (async_dada_writer && !dummy_run) {
      std::unique_lock<std::mutex> lock(dada_writer_mutex);
      dada_writer_cv.wait(lock, [this, gpu_buffer_index] {
        return gpu_buffer_free[gpu_buffer_index] || dada_writer_error != nullptr;
      });
      if (dada_writer_error)
        std::rethrow_exception(dada_writer_error);
      gpu_buffer_free[gpu_buffer_index] = false;
    }
    auto &b = buffers[gpu_buffer_index];
    // Re-steer tracked beams if due -- inert no-op when steering is disabled
    // (no --targets-filename). A due refresh enqueues the new weights onto
    // *every* buffer's stream in this one call, so all buffers always run
    // with identical weights. Must run here, before anything below reads
    // b.weights, and only from this single-threaded pipeline_feeder context
    // -- see the BeamSteering<T> comment block (pipeline.hpp above) for why
    // that ordering is what makes this safe without extra synchronization.
    beam_steering_.maybe_refresh();

    const uint64_t start_seq_num = packet_data->start_seq_id;
    const uint64_t end_seq_num = packet_data->end_seq_id;
    INFO_LOG("Pipeline run started with start_seq {} and end seq {}",
             start_seq_num, end_seq_num);

    const char *timing_env = std::getenv("PULSAR_CUDA_STAGE_TIMING");
    const bool time_stages = T::NR_FINE_CHANNELS > 1 && !dummy_run &&
                             !async_dada_writer &&
                             dada_key != 0 && timing_env &&
                             timing_env[0] == '1';
    cudaEvent_t stage_events[11]{};
    if (time_stages) {
      for (auto &event : stage_events)
        CUDA_CHECK(cudaEventCreate(&event));
      CUDA_CHECK(cudaEventRecord(stage_events[0], b.stream));
    }

    cudaEvent_t copy_done_event = nullptr;
    if (!dummy_run) {
      if (host_buffer_copy_done.empty()) {
        const size_t count = this->state_->input_buffer_count();
        if (count == 0)
          throw std::logic_error(
              "ProcessorState did not report its host input-buffer count");
        host_buffer_copy_done.resize(count, nullptr);
        for (auto &event : host_buffer_copy_done)
          CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        host_buffer_release_count = count;
        host_buffer_release_pending =
            std::make_unique<std::atomic<bool>[]>(count);
        for (size_t i = 0; i < count; ++i)
          host_buffer_release_pending[i].store(false,
                                               std::memory_order_relaxed);
        buffer_reclaimer_running.store(true, std::memory_order_release);
        buffer_reclaimer_thread = std::thread([this] {
          while (buffer_reclaimer_running.load(std::memory_order_acquire)) {
            bool found_pending = false;
            for (size_t i = 0; i < host_buffer_release_count; ++i) {
              if (!host_buffer_release_pending[i].load(
                      std::memory_order_acquire))
                continue;
              found_pending = true;
              const cudaError_t status =
                  cudaEventQuery(host_buffer_copy_done[i]);
              if (status == cudaSuccess) {
                host_buffer_release_pending[i].store(
                    false, std::memory_order_release);
                this->state_->release_buffer(static_cast<int>(i));
              } else if (status != cudaErrorNotReady) {
                std::cerr << "CUDA buffer reclaimer event query failed for host buffer "
                          << i << ": " << cudaGetErrorString(status) << std::endl;
                buffer_reclaimer_running.store(false,
                                                std::memory_order_release);
                break;
              }
            }
            if (!found_pending)
              std::this_thread::sleep_for(std::chrono::microseconds(50));
            else
              _mm_pause();
          }
        });
        INFO_LOG("Pulsar ingest created {} per-host-buffer H2D completion events",
                 count);
      }
      const int host_buffer_index = packet_data->buffer_index;
      if (host_buffer_index < 0 ||
          static_cast<size_t>(host_buffer_index) >= host_buffer_copy_done.size())
        throw std::out_of_range("packet_data host buffer index is out of range");
      if (host_buffer_release_pending[host_buffer_index].load(
              std::memory_order_acquire))
        throw std::logic_error(
            "host packet buffer was resubmitted before its H2D completion");
      copy_done_event = host_buffer_copy_done[host_buffer_index];
    }

    LambdaPipelineIngest<T>::ingest_and_scale(
        this->state_, packet_data, b.stream, b.host_stream,
        b.samples_entry.get(), b.scales.get(), d_gains, b.samples_half.get(),
        dummy_run, copy_done_event, dummy_run,
        time_stages ? stage_events[1] : nullptr);
    if (time_stages)
      CUDA_CHECK(cudaEventRecord(stage_events[2], b.stream));
    if (!dummy_run)
      host_buffer_release_pending[packet_data->buffer_index].store(
          true, std::memory_order_release);

    tensor_16.runPermutation("packetToPreAlign", alpha,
                             (__half *)b.samples_half.get(),
                             (__half *)b.samples_pre_align.get(), b.stream);
    if (time_stages)
      CUDA_CHECK(cudaEventRecord(stage_events[3], b.stream));

    // Pre-channelization: samples_pre_align/samples_aligned/samples_reordered are all
    // HalfPacketSamplesType-family buffers, shaped by the raw FPGA/coarse channel count.
    apply_delays_launch((__half *)b.samples_pre_align.get(),
                        (__half *)b.samples_aligned.get(), d_subpacket_delays,
                        T::NR_RECEIVERS_PER_PACKET, T::NR_FPGA_SOURCES,
                        T::NR_PACKETS_FOR_CORRELATION, T::NR_POLARIZATIONS,
                        T::NR_FPGA_CHANNELS, T::NR_TIME_STEPS_PER_PACKET, b.stream);
    if (time_stages)
      CUDA_CHECK(cudaEventRecord(stage_events[4], b.stream));

    reorder_streams_launch<T::NR_FPGA_SOURCES, T::NR_PACKETS_FOR_CORRELATION,
                           T::NR_TIME_STEPS_PER_PACKET, T::NR_FPGA_CHANNELS,
                           T::NR_RECEIVERS_PER_PACKET, T::NR_POLARIZATIONS>(
        (__half *)b.samples_aligned.get(),
        (__half *)b.samples_reordered.get(),
        d_stream_perm_recv, d_stream_perm_pol, b.stream);
    if (time_stages)
      CUDA_CHECK(cudaEventRecord(stage_events[5], b.stream));

    if constexpr (T::NR_FINE_CHANNELS > 1) {
      reorder_to_filter_input<
          T::NR_FPGA_CHANNELS, T::NR_POLARIZATIONS, T::NR_RECEIVERS,
          T::NR_RECEIVERS_PER_PACKET, T::NR_TIME_STEPS_PER_PACKET,
          T::NR_PACKETS_FOR_CORRELATION, FineChannelizer<T>::NR_TAPS,
          T::NR_FINE_CHANNELS>((__half *)b.samples_reordered.get(),
                               (float2 *)b.channelizer_input.get(), b.stream);

      if (time_stages)
        CUDA_CHECK(cudaEventRecord(stage_events[6], b.stream));

      channelizer_->launchAsync(b.stream, b.channelizer_input.get(),
                                b.channelizer_output.get());

      if (time_stages)
        CUDA_CHECK(cudaEventRecord(stage_events[7], b.stream));

      channelizer_output_to_col_maj_cons<
          T::NR_FPGA_CHANNELS, T::NR_FINE_CHANNELS, T::NR_FINE_CHANNEL_EDGE_TRIM,
          T::NR_POLARIZATIONS, T::NR_RECEIVERS, T::NR_RECEIVERS_PER_PACKET,
          FineChannelizer<T>::NR_SAMPLES_PER_FINE_CHANNEL,
          FineChannelizer<T>::NR_TIMES_PER_OUTPUT_BLOCK>(
          (const __half2 *)b.channelizer_output.get(),
          (__half *)b.samples_consolidated_col_maj.get(), b.stream,
          channelizer_->deripple_gains());
      if (time_stages)
        CUDA_CHECK(cudaEventRecord(stage_events[8], b.stream));

      if constexpr (RFI_MITIGATE) {
        channelizer_output_to_corr_input<
            T::NR_FPGA_CHANNELS, T::NR_FINE_CHANNELS, T::NR_FINE_CHANNEL_EDGE_TRIM,
            T::NR_POLARIZATIONS, T::NR_RECEIVERS, T::NR_PADDED_RECEIVERS,
            NR_BLOCKS_FOR_CORRELATION, NR_TIMES_PER_BLOCK>(
            (const __half2 *)b.channelizer_output.get(),
            (__half *)b.correlator_input.get(), b.stream,
            channelizer_->deripple_gains());
      }
    } else {
      tensor_16.runPermutation("alignedToPlanar", alpha,
                               (__half *)b.samples_reordered.get(),
                               (__half *)b.samples_consolidated.get(), b.stream);

      tensor_16.runPermutation(
          "consToColMajCons", alpha, (__half *)b.samples_consolidated.get(),
          (__half *)b.samples_consolidated_col_maj.get(), b.stream);
    }

    if (RFI_MITIGATE) {
      if constexpr (T::NR_FINE_CHANNELS == 1) {
        tensor_16.runPermutation(
            "alignedToPadding", alpha,
            reinterpret_cast<__half *>(b.samples_reordered.get()),
            reinterpret_cast<__half *>(b.samples_padding.get()), b.stream);

        CUDA_CHECK(
            cudaMemcpyAsync(b.samples_padded.get(), b.samples_padding.get(),
                            sizeof(typename T::HalfPacketAlignedSamplesType),
                            cudaMemcpyDefault, b.stream));
        CUDA_CHECK(
            cudaMemsetAsync(reinterpret_cast<char *>(b.samples_padded.get()) +
                                sizeof(typename T::HalfPacketAlignedSamplesType),
                            0,
                            sizeof(typename T::PaddedPacketSamplesType) -
                                sizeof(typename T::HalfPacketAlignedSamplesType),
                            b.stream));

        tensor_16.runPermutation(
            "paddedToCorrInput", alpha,
            reinterpret_cast<__half *>(b.samples_padded.get()),
            reinterpret_cast<__half *>(b.correlator_input.get()), b.stream);
      }

      // ------------------------------------------------------------------
      // 7. Cross-correlate with tcc::Correlator
      // ------------------------------------------------------------------
      correlator.launchAsync(
          static_cast<CUstream>(b.stream),
          reinterpret_cast<CUdeviceptr>(b.correlator_output.get()),
          reinterpret_cast<CUdeviceptr>(b.correlator_input.get()));
      // ------------------------------------------------------------------
      // 8. Rearrange correlator output to baseline-major, then trim padding
      // ------------------------------------------------------------------
      tensor_32.runPermutation(
          "visCorrToBaseline", alpha_32,
          reinterpret_cast<float *>(b.correlator_output.get()),
          reinterpret_cast<float *>(b.visibilities_baseline.get()), b.stream);

      CUDA_CHECK(cudaMemcpyAsync(
          b.visibilities_trimmed_baseline.get(), b.visibilities_baseline.get(),
          sizeof(TrimmedVisibilities) / 2, cudaMemcpyDefault, b.stream));

      void *source_pol_1_1 =
          (char *)b.visibilities_baseline.get() + 3 * sizeof(Visibilities) / 4;
      void *dest_pol_1_1 = (char *)b.visibilities_trimmed_baseline.get() +
                           sizeof(TrimmedVisibilities) / 2;

      CUDA_CHECK(cudaMemcpyAsync(dest_pol_1_1, source_pol_1_1,
                                 sizeof(TrimmedVisibilities) / 2,
                                 cudaMemcpyDefault, b.stream));
      tensor_32.runPermutation(
          "visBaselineTrimmedToDecomp", alpha_32,
          reinterpret_cast<float *>(b.visibilities_trimmed_baseline.get()),
          reinterpret_cast<float *>(b.visibilities_trimmed.get()), b.stream);

      unpack_triangular_baseline_batch_launch<cuComplex>(
          reinterpret_cast<cuComplex *>(b.visibilities_trimmed.get()),
          reinterpret_cast<cuComplex *>(b.decomp_visibilities.get()),
          T::NR_RECEIVERS, CUSOLVER_BATCH_SIZE, T::NR_CHANNELS, b.stream);

      CUSOLVER_CHECK(cusolverDnXsyevBatched(
          b.cusolver_handle, b.cusolver_params, cusolver_jobz, cusolver_uplo,
          T::NR_RECEIVERS, CUDA_C_32F,
          reinterpret_cast<void *>(b.decomp_visibilities.get()),
          T::NR_RECEIVERS, CUDA_R_32F,
          reinterpret_cast<void *>(b.eigenvalues.get()), CUDA_C_32F,
          b.cusolver_work_device.get(), b.cusolver_work_device_size,
          b.cusolver_work_host, b.cusolver_work_host_size,
          b.cusolver_info.get(), CUSOLVER_BATCH_SIZE));

      // ------------------------------------------------------------------
      // 11. Form P_block = U U^H via batched cuBLAS cherk.
      //
      //     After cuSOLVER, decomp_visibilities holds the full eigenvector
      //     matrix V (NR_RECEIVERS × NR_RECEIVERS, column = eigenvector,
      //     ascending order).  The signal subspace U consists of the last
      //     NR_SIGNAL_EIGENVECTORS columns, i.e. the sub-matrix starting at
      //     column offset (NR_RECEIVERS - NR_SIGNAL_EIGENVECTORS).
      //
      //     cuSOLVER stores column-major (Fortran order), so column j starts
      //     at row-offset 0 and the pointer to column j is:
      //       V_ptr + j * NR_RECEIVERS    (in cuComplex elements)
      //
      //     cherk computes:  C ← alpha * A * A^H + beta * C
      //       A = U  (NR_RECEIVERS × NR_SIGNAL_EIGENVECTORS, col-major)
      //       C = P  (NR_RECEIVERS × NR_RECEIVERS, col-major)
      //
      //     We call it once per batch element in a simple loop.  A batched
      //     cherk variant is not available in cuBLAS; the loop is over
      //     CUSOLVER_BATCH_SIZE elements and is negligible CPU overhead
      //     compared with the GPU kernels.
      // ------------------------------------------------------------------
      {
        constexpr int N = T::NR_RECEIVERS;
        const cuComplex herk_alpha{1.0f, 0.0f};
        const cuComplex herk_beta{0.0f, 0.0f}; // overwrite projection_block

        auto *V_base =
            reinterpret_cast<cuComplex *>(b.decomp_visibilities.get());
        auto *P_base =
            reinterpret_cast<cuComplex *>(b.float_projection_matrix.get());
        const size_t CUBLAS_BATCH_SIZE_PER_CHANNEL = T::NR_POLARIZATIONS;

        for (int channel = 0; channel < T::NR_CHANNELS; ++channel) {
          const int K = NR_SIGNAL_EIGENVECTORS[min_freq_channel + channel];
          const int col_offset = N - K; // first signal-subspace column
          // Pointer to signal-subspace U = last K columns of V_batch.
          for (int batch = 0; batch < CUBLAS_BATCH_SIZE_PER_CHANNEL; batch++) {
            // Pointer to the start of eigenvector matrix for this batch
            // element.
            cuComplex *V_batch =
                V_base +
                (channel * CUBLAS_BATCH_SIZE_PER_CHANNEL + batch) * N * N;
            cuComplex *U = V_batch + col_offset * N;
            // Pointer to output P for this batch element.
            cuComplex *P_batch =
                P_base +
                (channel * CUBLAS_BATCH_SIZE_PER_CHANNEL + batch) * N * N;

            CUBLAS_CHECK(cublasGemmEx(b.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_C,
                                      N, N, K, &herk_alpha, U, CUDA_C_32F, N, U,
                                      CUDA_C_32F, N, &herk_beta, P_batch,
                                      CUDA_C_32F, N, CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
          }
        }
      }

      computeIdentityMinusA((float2 *)b.float_projection_matrix.get(),
                            (__half2 *)b.projection_matrix.get(),
                            T::NR_RECEIVERS,
                            T::NR_CHANNELS * T::NR_POLARIZATIONS, b.stream);

      // conjugateMatrix((__half2 *)b.projection_matrix.get(),
      //                 T::NR_RECEIVERS * T::NR_RECEIVERS * T::NR_CHANNELS *
      //                     T::NR_POLARIZATIONS,
      //                 b.stream);

      {
        size_t CUBLAS_STRIDE_A = T::NR_RECEIVERS * T::NR_RECEIVERS;
        size_t CUBLAS_STRIDE_B = T::NR_RECEIVERS * T::NR_BEAMS;
        size_t CUBLAS_STRIDE_C = T::NR_RECEIVERS * T::NR_BEAMS;

        b.gemm_weight_projection_handle->Run(
            (CUdeviceptr)b.weights.get(),
            (CUdeviceptr)b.projection_matrix.get(),
            (CUdeviceptr)b.weights_updated.get());
      }

      // weightsDebugLaunch((__half2 *)b.weights_updated.get(),
      //                    T::NR_CHANNELS * T::NR_POLARIZATIONS *
      //                    T::NR_RECEIVERS
      //                    *
      //                        T::NR_BEAMS,
      //                    b.stream);

      tensor_16.runPermutation("weightsToBeamMajor", alpha,
                               (__half *)b.weights_updated.get(),
                               (__half *)b.weights_permuted.get(), b.stream);

      void *dest_ptr =
          (char *)b.weights_rfi_mitigated.get() + sizeof(BeamWeights);
      cudaMemcpyAsync(dest_ptr, b.weights_permuted.get(), sizeof(BeamWeights),
                      cudaMemcpyDefault, b.stream);

      tensor_16.runPermutation("weights2xBeamMajorToCCGLIB", alpha,
                               (__half *)b.weights_rfi_mitigated.get(),
                               (__half *)b.weights_beamformer.get(), b.stream);
    } else {
      // Re-permute b.weights -> b.weights_permuted on every execute_pipeline
      // call so that tracking updates written by maybe_refresh() (which lands
      // in b.weights) are reflected in the GEMM.  Without this step, the GEMM
      // always ran on the weights_permuted that was initialised once in the
      // constructor, ignoring every steering update after that.
      tensor_16.runPermutation("weightsToBeamMajor", alpha,
                               (__half *)b.weights.get(),
                               (__half *)b.weights_permuted.get(), b.stream);
      tensor_16.runPermutation("weights2xBeamMajorToCCGLIB", alpha,
                               (__half *)b.weights_permuted.get(),
                               (__half *)b.weights_beamformer.get(), b.stream);
    }

    b.gemm_handle->Run((CUdeviceptr)b.weights_beamformer.get(),
                       (CUdeviceptr)b.samples_consolidated_col_maj.get(),
                       (CUdeviceptr)b.beamformer_output.get());

    tensor_32.runPermutation("beamCCGLIBtoOutput", alpha_32,
                             (float *)b.beamformer_output.get(),
                             (float *)b.beam_output.get(), b.stream);
    if (time_stages)
      CUDA_CHECK(cudaEventRecord(stage_events[9], b.stream));

    if (beam_debug()) {
      // Copy the full beam output to host and report min/max/mean so we can
      // tell immediately whether the data is all-zeros or has real signal.
      // BEAM_DEBUG=1 gates this; it inserts a stream sync so don't leave on in
      // production.
      CUDA_CHECK(cudaStreamSynchronize(b.stream));
      constexpr size_t N = sizeof(BeamOutput) / sizeof(float);
      std::vector<float> h_beam(N);
      CUDA_CHECK(cudaMemcpy(h_beam.data(), b.beam_output.get(),
                            sizeof(BeamOutput), cudaMemcpyDefault));
      float bmin = h_beam[0], bmax = h_beam[0], bsum = 0.0f;
      int n_nonzero = 0;
      for (float v : h_beam) {
        if (v < bmin) bmin = v;
        if (v > bmax) bmax = v;
        bsum += v;
        if (v != 0.0f) ++n_nonzero;
      }
      std::cout << "[PulsarFold] beam_output: N=" << N
                << " min=" << bmin << " max=" << bmax
                << " mean=" << (bsum / static_cast<float>(N))
                << " nonzero=" << n_nonzero << "/" << N
                << " first4=[" << h_beam[0] << "," << h_beam[1]
                << "," << h_beam[2] << "," << h_beam[3] << "]\n";
    }

    if (output_ != nullptr && !dummy_run) {

      // PSRDADA beam streaming -- the header/data block writes below all
      // dereference `hdu`, which is null when the sink is disabled
      // (dada_key == 0). Skip them in that case; the eigen-output block that
      // follows uses `output_` (not PSRDADA) and is intentionally left
      // outside this guard.
      if (dada_key != 0) {
        constexpr size_t nr_dada_scalars =
            sizeof(DadaBeamOutput) / sizeof(float);
        constexpr int threads = 256;
        constexpr int blocks =
            static_cast<int>((nr_dada_scalars + threads - 1) / threads);
        if constexpr (DETECTED_INTENSITY) {
          detect_intensity_and_compact_shared_fine_boundaries_kernel<
              T::NR_CHANNELS, NR_DADA_CHANNELS,
              T::NR_EFFECTIVE_FINE_CHANNELS, NR_TIME_STEPS_PER_FINE_CHANNEL,
              T::NR_POLARIZATIONS, num_beams>
              <<<blocks, threads, 0, b.stream>>>(
                  reinterpret_cast<const float *>(b.beam_output.get()),
                  reinterpret_cast<float *>(dada_beam_outputs_[gpu_buffer_index].get()));
        } else {
          compact_shared_fine_boundaries_kernel<
              T::NR_CHANNELS, NR_DADA_CHANNELS,
              T::NR_EFFECTIVE_FINE_CHANNELS, NR_TIME_STEPS_PER_FINE_CHANNEL,
              T::NR_POLARIZATIONS, num_beams>
              <<<blocks, threads, 0, b.stream>>>(
                  reinterpret_cast<const float *>(b.beam_output.get()),
                  reinterpret_cast<float *>(dada_beam_outputs_[gpu_buffer_index].get()));
        }
        CUDA_CHECK(cudaGetLastError());

        if (async_dada_writer) {
          CUDA_CHECK(cudaEventRecord(dada_compute_done[gpu_buffer_index],
                                     b.stream));
          {
            std::lock_guard<std::mutex> lock(dada_writer_mutex);
            dada_ready_queue.push_back(gpu_buffer_index);
          }
          dada_writer_cv.notify_one();
        } else {

        uint64_t block_size = ipcbuf_get_bufsz((ipcbuf_t *)hdu->data_block);
        // write 1 block worth of data block via the "block" method
        {
          uint64_t rfi_block_size = 0;
          uint64_t block_id;
          char *block = ipcio_open_block_write(hdu->data_block, &block_id);
          if (!block) {
            multilog(log, LOG_ERR, "ipcio_open_block_write failed\n");
            std::cout << "open block write failed\n";
          }

          // control how much gets written using the block_size.
          // can toggle polarization from outer to inner dimensions
          // in order to control how many polarizations get written out.
          cudaMemcpyAsync(block, (char *)dada_beam_outputs_[gpu_buffer_index].get(), block_size,
                          cudaMemcpyDefault, b.stream);
          if (time_stages)
            CUDA_CHECK(cudaEventRecord(stage_events[10], b.stream));

          if constexpr (RFI_MITIGATE) {

            rfi_block_size = ipcbuf_get_bufsz((ipcbuf_t *)rfi_hdu->data_block);
            uint64_t rfi_block_id;
            char *rfi_block =
                ipcio_open_block_write(rfi_hdu->data_block, &rfi_block_id);
            if (!rfi_block) {
              multilog(log, LOG_ERR, "ipcio_open_block_write failed\n");
              std::cout << "open block write failed\n";
            }
            // This is a big hack it will only take the X pol right now.
            cudaMemcpyAsync(rfi_block,
                            (char *)dada_beam_outputs_[gpu_buffer_index].get() + block_size,
                            rfi_block_size, cudaMemcpyDefault, b.stream);
          }

          cudaDeviceSynchronize();

          if (time_stages) {
            static uint64_t timed_blocks = 0;
            static float stage_ms_total[10]{};
            for (int i = 0; i < 10; ++i) {
              float ms = 0.0f;
              CUDA_CHECK(cudaEventElapsedTime(&ms, stage_events[i],
                                              stage_events[i + 1]));
              stage_ms_total[i] += ms;
            }
            if (++timed_blocks % 16 == 0) {
              std::cout << "PULSAR_CUDA_STAGE_MS blocks=" << timed_blocks
                        << " h2d=" << stage_ms_total[0] / 16.0f
                        << " scale=" << stage_ms_total[1] / 16.0f
                        << " permute=" << stage_ms_total[2] / 16.0f
                        << " delay=" << stage_ms_total[3] / 16.0f
                        << " reorder=" << stage_ms_total[4] / 16.0f
                        << " filter_input=" << stage_ms_total[5] / 16.0f
                        << " filterbank=" << stage_ms_total[6] / 16.0f
                        << " filter_output=" << stage_ms_total[7] / 16.0f
                        << " beam=" << stage_ms_total[8] / 16.0f
                        << " detect_copy=" << stage_ms_total[9] / 16.0f
                        << std::endl;
              for (float &total : stage_ms_total)
                total = 0.0f;
            }
            for (auto &event : stage_events)
              CUDA_CHECK(cudaEventDestroy(event));
          }

          if (ipcio_close_block_write(hdu->data_block, block_size) < 0) {
            multilog(log, LOG_ERR, "ipcio_close_block_write failed\n");
          }
          if constexpr (RFI_MITIGATE) {
            if (ipcio_close_block_write(rfi_hdu->data_block, rfi_block_size) <
                0) {
              multilog(log, LOG_ERR, "ipcio_close_block_write failed\n");
            }
          }
        }
        }
      } // end PSRDADA beam streaming (dada_key != 0)

      if constexpr (RFI_MITIGATE) {
        size_t eig_block_num = output_->register_eigendecomposition_data_block(
            start_seq_num, end_seq_num);

        if (eig_block_num != std::numeric_limits<size_t>::max()) {
          void *eigval_ptr =
              output_->get_eigenvalues_data_landing_pointer(eig_block_num);
          void *eigvec_ptr =
              output_->get_eigenvectors_data_landing_pointer(eig_block_num);
          CUDA_CHECK(cudaMemcpyAsync(eigvec_ptr, b.decomp_visibilities.get(),
                                     sizeof(DecompositionVisibilities),
                                     cudaMemcpyDefault, b.stream));

          CUDA_CHECK(cudaMemcpyAsync(eigval_ptr, b.eigenvalues.get(),
                                     sizeof(Eigenvalues), cudaMemcpyDefault,
                                     b.stream));

          auto *ctx = new OutputTransferCompleteContext{
              .output = this->output_, .block_index = eig_block_num};
          CUDA_CHECK(cudaLaunchHostFunc(
              b.stream, eigen_output_transfer_complete_host_func, ctx));
        }
      }
    }

    if (async_dada_writer && !dummy_run)
      current_buffer = (current_buffer + 1) % num_buffers;
  }
  LambdaPulsarFoldPipeline(
      BeamWeightsT<T> *h_weights,
      const std::unordered_map<int, int> nr_signal_eigenvectors,
      const int min_freq_channel, key_t dada_key, std::string header_filename,
      key_t rfi_dada_key, BeamSteering<T> beam_steering,
      const CoarsePfbDeripple *deripple = nullptr)

      : num_buffers((dada_key != 0 && !RFI_MITIGATE) ? 3 : 1), h_weights(h_weights),
        beam_steering_(std::move(beam_steering)),
        correlator(cu::Device(0), tcc::Format::fp16, T::NR_PADDED_RECEIVERS,
                   T::NR_CHANNELS,
                   NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK,
                   T::NR_POLARIZATIONS, std::nullopt,
                   T::NR_PADDED_RECEIVERS_PER_BLOCK,
                   TCC_THREAD_BLOCKS_PER_SM),
        tensor_16(extent, CUTENSOR_R_16F, 128),
        tensor_32(extent, CUTENSOR_R_32F, 128),
        NR_SIGNAL_EIGENVECTORS(nr_signal_eigenvectors), header_written(false),
        min_freq_channel(min_freq_channel), dada_key(dada_key),
        async_dada_writer(dada_key != 0 && !RFI_MITIGATE),
        rfi_dada_key(rfi_dada_key) {
    std::cout << "Pulsar Fold instantiated with NR_CHANNELS: " << T::NR_CHANNELS
              << ", NR_RECEIVERS: " << T::NR_RECEIVERS
              << ", NR_POLARIZATIONS: " << T::NR_POLARIZATIONS
              << ", NR_SAMPLES_PER_CHANNEL: "
              << NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK
              << ", NR_TIMES_PER_BLOCK: " << NR_TIMES_PER_BLOCK
              << ", NR_BLOCKS_FOR_FFT: " << NR_BLOCKS_FOR_CORRELATION
              << ", NR_BEAMS: " << num_beams << std::endl;
    std::cout << "[PulsarFoldPipeline] beam output size is "
              << sizeof(BeamOutput) << " bytes." << std::endl;
    std::cout << "[PulsarFoldPipeline] PSRDADA output has "
              << NR_DADA_CHANNELS << " unique channels and block size "
              << sizeof(DadaBeamOutput) << " bytes." << std::endl;
    if (deripple) {
      if (deripple->voltage_gains.size() != T::NR_EFFECTIVE_FINE_CHANNELS)
        throw std::runtime_error("De-ripple table does not match retained fine bins");
      deripple_response_id_ = deripple->response_id;
    }

    const size_t NUM_TOTAL_BATCHES = num_beams * T::NR_CHANNELS *
                                     T::NR_POLARIZATIONS *
                                     T::NR_PACKETS_FOR_CORRELATION;

    // set up PSRDADA ring buffer. A zero dada_key disables the PSRDADA sink
    // entirely (no connect/lock, no header read, no block writes) -- the
    // pipeline then computes beams into b.beam_output but never streams them
    // out. Production always passes a real key (DADA_DEFAULT_BLOCK_KEY); the
    // zero-key path exists so tests can drive the full GPU compute without a
    // running ring buffer. Kept in lockstep with the same guard in
    // execute_pipeline's output block and in the destructor.
    log = nullptr;
    hdu = nullptr;
    rfi_hdu = nullptr;
    obs_header = nullptr;
    if (dada_key != 0) {
      log = multilog_open("pulsar_fold_writer", 0);
      multilog_add(log, stderr);
      hdu = dada_hdu_create(log);
      dada_hdu_set_key(hdu, dada_key);
      // connect to HDU
      if (dada_hdu_connect(hdu) < 0) {
        throw std::runtime_error("could not connect to PSRDADA HDU");
      }

      const uint64_t configured_block_size =
          ipcbuf_get_bufsz((ipcbuf_t *)hdu->data_block);
      constexpr uint64_t required_block_size =
          sizeof(DadaBeamOutput) / (RFI_MITIGATE ? 2 : 1);
      if (configured_block_size != required_block_size) {
        dada_hdu_disconnect(hdu);
        throw std::runtime_error(
            "PSRDADA data block size mismatch: ring has " +
            std::to_string(configured_block_size) + " bytes, pulsar-fold "
            "requires exactly " + std::to_string(required_block_size) +
            " bytes. Recreate only this ring with the required block size.");
      }
      std::cout << "[PulsarFoldPipeline] PSRDADA block size validated: "
                << configured_block_size << " bytes\n";

      // lock as writer on the HDU
      if (dada_hdu_lock_write(hdu) < 0) {
        multilog(log, LOG_ERR, "could not lock write on HDU\n");
      }

      if constexpr (RFI_MITIGATE) {
        rfi_hdu = dada_hdu_create(log);
        dada_hdu_set_key(rfi_hdu, rfi_dada_key);

        if (dada_hdu_connect(rfi_hdu) < 0) {
          multilog(log, LOG_ERR, "could not connect to RFI HDU\n");
        }

        // lock as writer on the HDU
        if (dada_hdu_lock_write(rfi_hdu) < 0) {
          multilog(log, LOG_ERR, "could not lock write on RFI HDU\n");
        }
      }

      obs_header = (char *)malloc(sizeof(char) * DADA_DEFAULT_HEADER_SIZE);

      if (fileread(header_filename.c_str(), obs_header,
                   DADA_DEFAULT_HEADER_SIZE) < 0) {
        free(obs_header);
        obs_header = nullptr;
        dada_hdu_unlock_write(hdu);
        dada_hdu_disconnect(hdu);
        throw std::runtime_error("could not read PSRDADA ASCII header from " +
                                 header_filename);
      }

      // Overwrite UTC_START with the current wall-clock UTC so DSPSR folds
      // at the correct pulsar phase. The header file contains a placeholder
      // that is only valid for the exact instant it was written.
      {
        time_t now = time(nullptr);
        struct tm utc_tm{};
        gmtime_r(&now, &utc_tm);
        char utc_start[32];
        strftime(utc_start, sizeof(utc_start), "%Y-%m-%d-%H:%M:%S", &utc_tm);
        if (ascii_header_set(obs_header, "UTC_START", "%s", utc_start) < 0)
          fprintf(stderr, "WARNING: could not set UTC_START in PSRDADA header\n");
        else
          fprintf(stderr, "INFO: PSRDADA header UTC_START set to %s\n", utc_start);
      }
      if (ascii_header_set(obs_header, "DERIPPLE", "%d",
                           deripple ? 1 : 0) < 0)
        throw std::runtime_error("Could not set DERIPPLE in PSRDADA header");
      if (deripple &&
          ascii_header_set(obs_header, "DRIPRID", "%s",
                           deripple_response_id_.c_str()) < 0)
        throw std::runtime_error("Could not set DRIPRID in PSRDADA header");

      // Publish metadata before the expensive channelizer/JIT warmup below.
      // DSPSR cannot build its coherent-dedispersion plan until it sees this
      // header (about 50 seconds for 1081 channels on this host). Publishing
      // on the first live data block instead filled the 64-block ring after
      // only 1.12 seconds and stalled capture while DSPSR was still preparing.
      const uint64_t header_size = ipcbuf_get_bufsz(hdu->header_block);
      if (header_size != DADA_DEFAULT_HEADER_SIZE)
        throw std::runtime_error(
            "PSRDADA header block size mismatch: expected " +
            std::to_string(DADA_DEFAULT_HEADER_SIZE) + ", ring has " +
            std::to_string(header_size));
      char *header = ipcbuf_get_next_write(hdu->header_block);
      if (!header)
        throw std::runtime_error("could not acquire PSRDADA header block");
      std::memcpy(header, obs_header, header_size);
      if (ipcbuf_mark_filled(hdu->header_block, header_size) < 0)
        throw std::runtime_error("could not publish PSRDADA header block");

      if constexpr (RFI_MITIGATE) {
        const uint64_t rfi_header_size =
            ipcbuf_get_bufsz(rfi_hdu->header_block);
        if (rfi_header_size != header_size)
          throw std::runtime_error("RFI PSRDADA header block size mismatch");
        char *rfi_header = ipcbuf_get_next_write(rfi_hdu->header_block);
        if (!rfi_header)
          throw std::runtime_error("could not acquire RFI PSRDADA header block");
        std::memcpy(rfi_header, obs_header, header_size);
        if (ipcbuf_mark_filled(rfi_hdu->header_block, rfi_header_size) < 0)
          throw std::runtime_error("could not publish RFI PSRDADA header block");
      }
      header_written = true;
      std::cout << "[PulsarFoldPipeline] PSRDADA header published before GPU warmup\n";
    }
    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modePacketPreAlign, "prealign");
    tensor_16.addTensor(modePacketPadding, "packet_padding");
    tensor_16.addTensor(modePacketAligned, "aligned");
    tensor_16.addTensor(modePacketPadded, "packet_padded");
    tensor_16.addTensor(modePlanar, "planar");
    tensor_16.addTensor(modePlanarCons, "planarCons");
    tensor_16.addTensor(modePlanarColMajCons, "planarColMajCons");

    tensor_16.addTensor(modeWeightsInput, "weightsInput");
    tensor_16.addTensor(modeWeightsBeamMajor, "weightsBeamMajor");
    tensor_16.addTensor(modeWeights2xBeamMajor, "weights2xBeamMajor");
    tensor_16.addTensor(modeWeightsCCGLIB, "weightsCCGLIB");

    tensor_32.addTensor(modeBeamCCGLIB, "beamCCGLIB");
    tensor_32.addTensor(modeBeamOutput, "beamOutput");

    tensor_16.addTensor(modeCorrelatorInput, "corr_input");
    tensor_32.addTensor(modeVisCorr, "visCorr");
    tensor_32.addTensor(modeVisCorrBaseline, "visBaseline");
    tensor_32.addTensor(modeVisCorrBaselineTrimmed, "visBaselineTrimmed");
    tensor_32.addTensor(modeVisDecomp, "visDecomp");

    // Permutation descriptors
    tensor_16.addPermutation("packet_padded", "corr_input",
                             CUTENSOR_COMPUTE_DESC_16F, "paddedToCorrInput");

    tensor_16.addPermutation("aligned", "packet_padding",
                             CUTENSOR_COMPUTE_DESC_16F, "alignedToPadding");
    tensor_16.addPermutation("packet", "prealign", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToPreAlign");
    tensor_16.addPermutation("aligned", "planar", CUTENSOR_COMPUTE_DESC_16F,
                             "alignedToPlanar");
    tensor_32.addPermutation("visCorr", "visBaseline",
                             CUTENSOR_COMPUTE_DESC_32F, "visCorrToBaseline");
    tensor_32.addPermutation("visBaselineTrimmed", "visDecomp",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "visBaselineTrimmedToDecomp");

    tensor_16.addPermutation("planarCons", "planarColMajCons",
                             CUTENSOR_COMPUTE_DESC_16F, "consToColMajCons");
    tensor_16.addPermutation("weightsInput", "weightsBeamMajor",
                             CUTENSOR_COMPUTE_DESC_16F, "weightsToBeamMajor");

    tensor_16.addPermutation("weights2xBeamMajor", "weightsCCGLIB",
                             CUTENSOR_COMPUTE_DESC_16F,
                             "weights2xBeamMajorToCCGLIB");

    tensor_32.addPermutation("beamCCGLIB", "beamOutput",
                             CUTENSOR_COMPUTE_DESC_32F, "beamCCGLIBtoOutput");

    CUDA_CHECK(cudaMalloc((void **)&d_subpacket_delays,
                          sizeof(int) * T::NR_FPGA_SOURCES));
    CUDA_CHECK(cudaMalloc((void **)&d_gains, sizeof(typename T::AntennaGains)));

    CUDA_CHECK(
        cudaMemset(d_subpacket_delays, 0, sizeof(int) * T::NR_FPGA_SOURCES));

    auto default_gains = get_default_gains<T::NR_FPGA_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));

    {
      constexpr int NR_RECVS = T::NR_FPGA_SOURCES * T::NR_RECEIVERS_PER_PACKET;
      constexpr int NR_PERM  = NR_RECVS * (int)T::NR_POLARIZATIONS;
      CUDA_CHECK(cudaMalloc((void **)&d_stream_perm_recv, sizeof(int) * NR_PERM));
      CUDA_CHECK(cudaMalloc((void **)&d_stream_perm_pol,  sizeof(int) * NR_PERM));
      std::vector<int> identity_recv(NR_PERM), identity_pol(NR_PERM);
      for (int i = 0; i < NR_RECVS; ++i)
        for (int p = 0; p < (int)T::NR_POLARIZATIONS; ++p) {
          identity_recv[i * T::NR_POLARIZATIONS + p] = i;
          identity_pol [i * T::NR_POLARIZATIONS + p] = p;
        }
      CUDA_CHECK(cudaMemcpy(d_stream_perm_recv, identity_recv.data(),
                            sizeof(int) * NR_PERM, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_stream_perm_pol,  identity_pol.data(),
                            sizeof(int) * NR_PERM, cudaMemcpyHostToDevice));
    }

    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);
    if constexpr (T::NR_FINE_CHANNELS > 1) {
      channelizer_ = std::make_unique<FineChannelizer<T>>(cu_device);
    }
    buffers.reserve(num_buffers);
    dada_beam_outputs_.reserve(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(cu_device);
      dada_beam_outputs_.push_back(make_device_ptr<DadaBeamOutput>());

      // Finalize cuFFT plan for this buffer
      auto &b = buffers.back();

      // Copy initial weights
      cudaMemcpyAsync(b.weights.get(), h_weights, sizeof(BeamWeights),
                      cudaMemcpyDefault, b.stream);
      tensor_16.runPermutation("weightsToBeamMajor", alpha,
                               (__half *)b.weights.get(),
                               (__half *)b.weights_permuted.get(), b.stream);
      if constexpr (RFI_MITIGATE) {
        cudaMemcpyAsync(b.weights_rfi_mitigated.get(), b.weights_permuted.get(),
                        sizeof(BeamWeights), cudaMemcpyDefault, b.stream);
      }
      // Registered before the warmup run below, so the first (always-overdue)
      // maybe_refresh() steers every buffer in one shot.
      beam_steering_.register_buffer(b.weights.get(), b.stream);
    }
    last_frame_processed = 0;
    current_buffer = 0;
    cudaDeviceSynchronize();
    // warm up the pipeline.
    // This will JIT the template kernels to avoid having a long startup time
    // Because everything is zeroed it should have negligible effect on output.
    typename T::PacketFinalDataType warmup_packet;
    std::memset(warmup_packet.samples, 0,
                warmup_packet.get_samples_elements_size());
    std::memset(warmup_packet.scales, 0,
                warmup_packet.get_scales_element_size());
    std::memset(warmup_packet.arrivals, 0, warmup_packet.get_arrivals_size());
    execute_pipeline(&warmup_packet, true);
    cudaDeviceSynchronize();
    if (async_dada_writer) {
      gpu_buffer_free.assign(num_buffers, true);
      dada_compute_done.resize(num_buffers, nullptr);
      for (auto &event : dada_compute_done)
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      CUDA_CHECK(cudaStreamCreateWithFlags(&dada_copy_stream,
                                           cudaStreamNonBlocking));
      dada_writer_thread = std::thread([this] { run_dada_writer(); });
    }
  };

  ~LambdaPulsarFoldPipeline() {
    if (async_dada_writer) {
      {
        std::lock_guard<std::mutex> lock(dada_writer_mutex);
        dada_writer_stopping = true;
      }
      dada_writer_cv.notify_all();
      if (dada_writer_thread.joinable())
        dada_writer_thread.join();
      if (dada_writer_error)
        std::cerr << "Pulsar DADA writer failed before shutdown" << std::endl;
      for (auto event : dada_compute_done)
        if (event) cudaEventDestroy(event);
      if (dada_copy_stream)
        cudaStreamDestroy(dada_copy_stream);
    }
    buffer_reclaimer_running.store(false, std::memory_order_release);
    if (buffer_reclaimer_thread.joinable())
      buffer_reclaimer_thread.join();
    for (auto &b : buffers) {
      if (b.stream)
        CUDA_CHECK(cudaStreamSynchronize(b.stream));
      if (b.host_stream)
        CUDA_CHECK(cudaStreamSynchronize(b.host_stream));
    }
    for (auto event : host_buffer_copy_done) {
      if (event)
        CUDA_CHECK(cudaEventDestroy(event));
    }
    if (d_stream_perm_recv) cudaFree(d_stream_perm_recv);
    if (d_stream_perm_pol)  cudaFree(d_stream_perm_pol);
    if (obs_header) {
      free(obs_header);
      obs_header = nullptr;
    }
    // Mirror the constructor: nothing to tear down when the PSRDADA sink was
    // disabled (dada_key == 0), and hdu/log are left null in that case.
    if (dada_key == 0)
      return;

    // Signal end-of-data on the data block before releasing the write lock.
    // Without this, DSPSR's reader loop blocks indefinitely on the next
    // ipcio_open_block_read() call -- it sees a broken writer connection
    // rather than a clean observation end, and does not finalize the folded
    // profile.
    if (ipcbuf_enable_eod((ipcbuf_t *)hdu->data_block) < 0)
      multilog(log, LOG_ERR, "ipcbuf_enable_eod failed on data block\n");

    if (dada_hdu_unlock_write(hdu) < 0) {
      multilog(log, LOG_ERR, "dada_hdu_unlock_write failed\n");
    }

    // disconnect from HDU
    if (dada_hdu_disconnect(hdu) < 0)
      multilog(log, LOG_ERR, "could not disconnect from hdu\n");

    if constexpr (RFI_MITIGATE) {
      if (ipcbuf_enable_eod((ipcbuf_t *)rfi_hdu->data_block) < 0)
        multilog(log, LOG_ERR, "ipcbuf_enable_eod failed on rfi data block\n");

      if (dada_hdu_unlock_write(rfi_hdu) < 0) {
        multilog(log, LOG_ERR, "dada_hdu_unlock_write failed\n");
      }

      // disconnect from HDU
      if (dada_hdu_disconnect(rfi_hdu) < 0)
        multilog(log, LOG_ERR, "could not disconnect from rfi hdu\n");
    }
  }

  void dump_visibilities(const uint64_t end_seq_num = 0) override {
    if (async_dada_writer) {
      std::unique_lock<std::mutex> lock(dada_writer_mutex);
      dada_writer_cv.wait(lock, [this] {
        if (dada_writer_error)
          return true;
        if (!dada_ready_queue.empty())
          return false;
        for (bool free : gpu_buffer_free)
          if (!free)
            return false;
        return true;
      });
      if (dada_writer_error)
        std::rethrow_exception(dada_writer_error);
    }
  }

  virtual void set_subpacket_delays(int *delays_subpacket) {
    subpacket_delays_ = delays_subpacket;
    CUDA_CHECK(cudaMemcpy(d_subpacket_delays, subpacket_delays_,
                          sizeof(int) * T::NR_FPGA_SOURCES, cudaMemcpyDefault));
  }

  // Test/debug hook. After execute_pipeline() completes, copies the most
  // recently computed beamformer output -- the exact device-side
  // `b.beam_output` buffer that gets streamed to PSRDADA -- to a host
  // destination (`dst` must have room for beam_output_size_bytes()). Lets
  // tests inspect the beams without a PSRDADA sink (see the dada_key == 0
  // path); not used on the production hot path.
  void copy_latest_beam_output_to_host(void *dst) {
    auto &b = buffers[current_buffer];
    CUDA_CHECK(cudaStreamSynchronize(b.stream));
    CUDA_CHECK(cudaMemcpy(dst, b.beam_output.get(), sizeof(BeamOutput),
                          cudaMemcpyDefault));
  }
  static constexpr size_t beam_output_size_bytes() {
    return sizeof(BeamOutput);
  }
  // Beam-output shape: [NUM_BEAMS][NR_TIMES][NR_CHANNELS][NR_POL][COMPLEX].
  static constexpr int beam_output_num_beams() { return num_beams; }
  static constexpr int beam_output_num_times() {
    return NR_TIME_STEPS_PER_FINE_CHANNEL;
  }
};
