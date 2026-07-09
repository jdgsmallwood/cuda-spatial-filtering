#pragma once

template <typename T>
class LambdaAdaptiveBeamformedSpectraPipeline : public GPUPipeline {
private:
  static constexpr int NR_TIMES_PER_BLOCK = 128 / 16; // NR_BITS;

  static constexpr int NR_BLOCKS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET /
      NR_TIMES_PER_BLOCK;
  static constexpr int NR_TIME_STEPS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET;
  static constexpr int COMPLEX = 2;

  static constexpr int NR_BASELINES =
      T::NR_PADDED_RECEIVERS * (T::NR_PADDED_RECEIVERS + 1) / 2;
  static constexpr int NR_UNPADDED_BASELINES =
      T::NR_RECEIVERS * (T::NR_RECEIVERS + 1) / 2;
  // cuSOLVER batch size: one (NR_RECEIVERS × NR_RECEIVERS) matrix per
  // channel × pol × pol, matching LambdaGPUPipeline exactly.
  static constexpr int CUSOLVER_BATCH_SIZE =
      T::NR_CHANNELS * T::NR_POLARIZATIONS;
  static constexpr int NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE = 5;

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

  using FFTCUFFTInputType =
      float2[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
            [T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION];
  using FFTCUFFTOutputType =
      float2[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
            [T::NR_PACKETS_FOR_CORRELATION][T::NR_TIME_STEPS_PER_PACKET];
  using FineChannelRemovedType =
      float2[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
            [T::NR_PACKETS_FOR_CORRELATION]
            [T::NR_TIME_STEPS_PER_PACKET -
             2 * NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE];
  using FineChannelCopyType =
      float2[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
            [T::NR_PACKETS_FOR_CORRELATION]
            [(T::NR_TIME_STEPS_PER_PACKET -
              2 * NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE) /
             2];
  using FineChannelSeekType =
      float2[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
            [T::NR_PACKETS_FOR_CORRELATION]
            [T::NR_TIME_STEPS_PER_PACKET / 2 +
             NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE];
  using FFTOutputType =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
           [T::NR_TIME_STEPS_PER_PACKET -
            2 * NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE];
  using BeamformerOutput =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
           [NR_TIME_STEPS_FOR_CORRELATION][COMPLEX];
  using BeamOutput =
      std::complex<__half>[T::NR_CHANNELS][T::NR_POLARIZATIONS][2 * T::NR_BEAMS]
                          [NR_TIME_STEPS_FOR_CORRELATION];
  using ProjectionMatrix =
      std::complex<__half>[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                          [T::NR_RECEIVERS];
  using FloatProjectionMatrix =
      std::complex<float>[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                         [T::NR_RECEIVERS];

  using BeamWeights = BeamWeightsT<T>;
  struct RFIMitigatedT {
    static constexpr size_t NR_CHANNELS = T::NR_CHANNELS;
    static constexpr size_t NR_POLARIZATIONS = T::NR_POLARIZATIONS;
    static constexpr size_t NR_BEAMS = 2 * T::NR_BEAMS;
    static constexpr size_t NR_RECEIVERS = T::NR_RECEIVERS;
  };

  using RFIMitigatedBeamWeights = BeamWeightsT<RFIMitigatedT>;

  struct PipelineResources {
    cudaStream_t stream;
    cudaStream_t host_stream;
    ManagedCufftPlan fft_plan, fft_plan_fine_channel;

    DevicePtr<typename T::InputPacketSamplesType> samples_entry;
    DevicePtr<typename T::PacketScalesType> scales;
    DevicePtr<typename T::HalfPacketSamplesType> samples_half,
        samples_pre_align;
    DevicePtr<typename T::HalfPacketAlignedSamplesType> samples_aligned,
        samples_consolidated, samples_consolidated_col_maj, samples_padding;
    DevicePtr<typename T::PaddedPacketSamplesType> samples_padded;
    DevicePtr<FFTCUFFTInputType> samples_cufft_input;
    DevicePtr<BeamOutput> beam_output;
    DevicePtr<FFTCUFFTOutputType> samples_cufft_output,
        samples_cufft_output_fine_channel;
    DevicePtr<FineChannelRemovedType> samples_fine_channel_removed,
        cufft_downsampled_input;
    DevicePtr<BeamformerOutput> beam_output_float;
    DevicePtr<FFTOutputType> cufft_downsampled_output;
    DevicePtr<BeamWeights> weights;
    DevicePtr<BeamWeights> weights_permuted, weights_updated;
    DevicePtr<RFIMitigatedBeamWeights> weights_rfi_mitigated;
    DevicePtr<RFIMitigatedBeamWeights> weights_beamformer;
    DevicePtr<BeamformerOutput> beamformer_output;
    DevicePtr<void> cufft_work_area;

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
    // Eigenvectors scaled column-wise by d[] (see execute_pipeline).
    DevicePtr<DecompositionVisibilities> scaled_eigenvectors;
    DevicePtr<ProjectionMatrix> projection_matrix;
    DevicePtr<FloatProjectionMatrix> float_projection_matrix;

    // Per-block eigenvalues (ascending order from cuSOLVER).
    DevicePtr<Eigenvalues> eigenvalues;
    // Device copy of per-eigenvector scale factors d[channel][pol][k].
    DevicePtr<Eigenvalues> d_scale_factors;
    // Per-batch signal-subspace counts, fixed and detected variants.
    DevicePtr<int32_t> d_fixed_detected_counts;
    DevicePtr<int32_t> d_detected_counts;
    int32_t *h_detected_counts_staging = nullptr;
    std::shared_ptr<std::mutex> eigenmode_stats_mutex =
        std::make_shared<std::mutex>();
    std::shared_ptr<std::vector<int32_t>> eigenmode_stats_history =
        std::make_shared<std::vector<int32_t>>();

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

    PipelineResources(CUdevice cu_device, size_t work_size)
        : samples_entry(make_device_ptr<typename T::InputPacketSamplesType>()),
          scales(make_device_ptr<typename T::PacketScalesType>()),
          samples_half(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_pre_align(
              make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_aligned(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_consolidated(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_consolidated_col_maj(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_padding(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_cufft_input(make_device_ptr<FFTCUFFTInputType>()),
          beam_output(make_device_ptr<BeamOutput>()),
          beam_output_float(make_device_ptr<BeamformerOutput>()),
          samples_cufft_output(make_device_ptr<FFTCUFFTOutputType>()),
          samples_cufft_output_fine_channel(
              make_device_ptr<FFTCUFFTOutputType>()),
          samples_fine_channel_removed(
              make_device_ptr<FineChannelRemovedType>()),
          cufft_downsampled_output(make_device_ptr<FFTOutputType>()),
          cufft_downsampled_input(make_device_ptr<FineChannelRemovedType>()),
          weights(make_device_ptr<BeamWeights>()),
          weights_permuted(make_device_ptr<BeamWeights>()),
          weights_updated(make_device_ptr<BeamWeights>()),
          weights_rfi_mitigated(make_device_ptr<RFIMitigatedBeamWeights>()),
          weights_beamformer(make_device_ptr<RFIMitigatedBeamWeights>()),
          beamformer_output(make_device_ptr<BeamformerOutput>()),
          samples_padded(
              make_device_ptr<typename T::PaddedPacketSamplesType>()),
          correlator_input(make_device_ptr<CorrelatorInput>()),
          correlator_output(make_device_ptr<CorrelatorOutput>()),
          visibilities_baseline(make_device_ptr<Visibilities>()),
          visibilities_trimmed_baseline(make_device_ptr<TrimmedVisibilities>()),
          visibilities_trimmed(make_device_ptr<TrimmedVisibilities>()),
          decomp_visibilities(make_device_ptr<DecompositionVisibilities>()),
          scaled_eigenvectors(make_device_ptr<DecompositionVisibilities>()),
          projection_matrix(make_device_ptr<ProjectionMatrix>()),
          float_projection_matrix(make_device_ptr<FloatProjectionMatrix>()),
          eigenvalues(make_device_ptr<Eigenvalues>()),
          d_scale_factors(make_device_ptr<Eigenvalues>()),
          d_fixed_detected_counts(
              make_device_ptr<int32_t>(CUSOLVER_BATCH_SIZE *
                                       sizeof(int32_t))),
          d_detected_counts(make_device_ptr<int32_t>(CUSOLVER_BATCH_SIZE *
                                                     sizeof(int32_t))),
          cusolver_info(
              make_device_ptr<int>(CUSOLVER_BATCH_SIZE * sizeof(int))),
          cufft_work_area(make_device_ptr<void>(work_size)) {
      // Stream Creation
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUDA_CHECK(
          cudaStreamCreateWithFlags(&host_stream, cudaStreamNonBlocking));

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

      const std::complex<float> alpha_ccglib = {1, 0};
      const std::complex<float> beta_ccglib = {0, 0};
      // GEMM Initialization
      gemm_handle = std::make_unique<ccglib::pipeline::Pipeline>(
          T::NR_CHANNELS * T::NR_POLARIZATIONS, 2 * T::NR_BEAMS,
          NR_TIMES_PER_BLOCK * NR_BLOCKS_FOR_CORRELATION, T::NR_RECEIVERS,
          cu_device, stream, ccglib::complex_planar, ccglib::complex_planar,
          ccglib::mma::row_major, ccglib::mma::col_major,
          ccglib::mma::row_major, ccglib::ValueType::float16,
          ccglib::ValueType::float32, ccglib::mma::opt, alpha_ccglib,
          beta_ccglib);

      gemm_weight_projection_handle =
          std::make_unique<ccglib::pipeline::Pipeline>(
              T::NR_CHANNELS * T::NR_POLARIZATIONS, T::NR_BEAMS,
              T::NR_RECEIVERS, T::NR_RECEIVERS, cu_device, stream,
              ccglib::complex_interleaved, ccglib::complex_interleaved,
              ccglib::mma::row_major, ccglib::mma::col_major,
              ccglib::mma::row_major, ccglib::ValueType::float16,
              ccglib::ValueType::float16, ccglib::mma::opt, alpha_ccglib,
              beta_ccglib);
      CUBLAS_CHECK(cublasCreate(&cublas_handle));
      CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
      CUDA_CHECK(cudaMallocHost((void **)&h_detected_counts_staging,
                                CUSOLVER_BATCH_SIZE * sizeof(int32_t)));
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
      if (h_detected_counts_staging)
        cudaFreeHost(h_detected_counts_staging);
    }
    PipelineResources(PipelineResources &&other) noexcept
        : stream(other.stream), host_stream(other.host_stream),
          fft_plan(std::move(other.fft_plan)),
          samples_entry(std::move(other.samples_entry)),
          scales(std::move(other.scales)),
          samples_half(std::move(other.samples_half)),
          samples_consolidated(std::move(other.samples_consolidated)),
          samples_consolidated_col_maj(
              std::move(other.samples_consolidated_col_maj)),
          beam_output(std::move(other.beam_output)),
          beam_output_float(std::move(other.beam_output_float)),
          samples_cufft_input(std::move(other.samples_cufft_input)),
          samples_cufft_output(std::move(other.samples_cufft_output)),
          cufft_downsampled_output(std::move(other.cufft_downsampled_output)),
          weights(std::move(other.weights)),
          weights_permuted(std::move(other.weights_permuted)),
          beamformer_output(std::move(other.beamformer_output)),
          cufft_work_area(std::move(other.cufft_work_area)),
          gemm_handle(std::move(other.gemm_handle)),
          samples_padding(std::move(other.samples_padding)),
          samples_padded(std::move(other.samples_padded)),
          correlator_input(std::move(other.correlator_input)),
          correlator_output(std::move(other.correlator_output)),
          float_projection_matrix(std::move(other.float_projection_matrix)),
          projection_matrix(std::move(other.projection_matrix)),
          d_scale_factors(std::move(other.d_scale_factors)),
          d_fixed_detected_counts(std::move(other.d_fixed_detected_counts)),
          d_detected_counts(std::move(other.d_detected_counts)),
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
          cusolver_info(std::move(other.cusolver_info)),
          h_detected_counts_staging(other.h_detected_counts_staging),
          eigenmode_stats_mutex(std::move(other.eigenmode_stats_mutex)),
          eigenmode_stats_history(std::move(other.eigenmode_stats_history))

    {
      other.stream = nullptr;
      other.host_stream = nullptr;
      other.cusolver_handle = nullptr;
      other.cusolver_params = nullptr;
      other.cusolver_work_host = nullptr;
      other.h_detected_counts_staging = nullptr;
    }

    PipelineResources &operator=(PipelineResources &&other) noexcept {
      if (this != &other) {
        if (stream)
          cudaStreamDestroy(stream);
        if (host_stream)
          cudaStreamDestroy(host_stream);

        stream = other.stream;
        host_stream = other.host_stream;
        fft_plan = std::move(other.fft_plan);
        samples_entry = std::move(other.samples_entry);
        scales = std::move(other.scales);
        samples_half = std::move(other.samples_half);
        samples_consolidated = std::move(other.samples_consolidated);
        samples_consolidated_col_maj =
            std::move(other.samples_consolidated_col_maj);
        beam_output = std::move(other.beam_output);
        beam_output_float = std::move(other.beam_output_float);
        samples_cufft_input = std::move(other.samples_cufft_input);
        samples_cufft_output = std::move(other.samples_cufft_output);
        cufft_downsampled_output = std::move(other.cufft_downsampled_output);
        weights = std::move(other.weights);
        weights_permuted = std::move(other.weights_permuted);
        beamformer_output = std::move(other.beamformer_output);
        cufft_work_area = std::move(other.cufft_work_area);
        gemm_handle = std::move(other.gemm_handle);
        samples_padding = std::move(other.samples_padding);
        samples_padded = std::move(other.samples_padded);
        correlator_input = std::move(other.correlator_input);
        correlator_output = std::move(other.correlator_output);
        float_projection_matrix = std::move(other.float_projection_matrix);
        projection_matrix = std::move(other.projection_matrix);
        d_scale_factors = std::move(other.d_scale_factors);
        d_fixed_detected_counts = std::move(other.d_fixed_detected_counts);
        d_detected_counts = std::move(other.d_detected_counts);
        visibilities_baseline = std::move(other.visibilities_baseline);
        visibilities_trimmed_baseline =
            std::move(other.visibilities_trimmed_baseline);
        visibilities_trimmed = std::move(other.visibilities_trimmed);
        decomp_visibilities = std::move(other.decomp_visibilities);
        eigenvalues = std::move(other.eigenvalues);
        cusolver_handle = other.cusolver_handle;
        cusolver_params = other.cusolver_params;
        cusolver_work_device = std::move(other.cusolver_work_device);
        cusolver_work_host = other.cusolver_work_host;
        cusolver_work_device_size = other.cusolver_work_device_size;
        cusolver_work_host_size = other.cusolver_work_host_size;
        cusolver_info = std::move(other.cusolver_info);
        h_detected_counts_staging = other.h_detected_counts_staging;
        eigenmode_stats_mutex = std::move(other.eigenmode_stats_mutex);
        eigenmode_stats_history = std::move(other.eigenmode_stats_history);

        other.stream = nullptr;

        other.host_stream = nullptr;
        other.cusolver_handle = nullptr;
        other.cusolver_params = nullptr;
        other.cusolver_work_host = nullptr;
        other.h_detected_counts_staging = nullptr;
      }
      return *this;
    }

    PipelineResources(const PipelineResources &) = delete;
    PipelineResources &operator=(const PipelineResources &) = delete;
  };

  int num_buffers;
  std::vector<PipelineResources> buffers;

  cusolverEigMode_t cusolver_jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t cusolver_uplo = CUBLAS_FILL_MODE_UPPER;
  tcc::Correlator correlator;

  inline static const __half alpha = __float2half(1.0f);

  std::unordered_map<int, int> NR_SIGNAL_EIGENVECTORS;
  int min_freq_channel;
  bool shrink_eigenvalues_;
  bool detect_signal_eigenmodes_;
  float detection_threshold_delta_;
  double eigenmode_stats_interval_seconds_;

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

  inline static const std::vector<int> modePacket{'c', 'y', 'f', 'u',
                                                  'n', 'p', 'z'};
  inline static const std::vector<int> modePlanar{'c', 'p', 'z', 'f',
                                                  'n', 'o', 'u'};

  inline static const std::vector<int> modePacketPreAlign{'f', 'y', 'u', 'c',
                                                          'n', 'p', 'z'};
  inline static const std::vector<int> modePacketAligned{'f', 'o', 'u', 'c',
                                                         'n', 'p', 'z'};
  inline static const std::vector<int> modeCUFFTInput{'c', 'p', 'e', 's', 'z'};
  inline static const std::vector<int> modeCUFFTOutput{'c', 'p', 'e',
                                                       'o', 'u', 'z'};
  inline static const std::vector<int> modeFineChannelRemove{'u', 'c', 'p',
                                                             'e', 'o', 'z'};
  inline static const std::vector<int> modeFineChannelRemoved{'g', 'c', 'p',
                                                              'e', 'o', 'z'};
  inline static const std::vector<int> modeBeamFFTDownsample{'c', 'p', 'e',
                                                             'o', 'g', 'z'};
  // We need the planar samples matrix to be in column-major memory layout
  // which is equivalent to transposing time and receiver structure here.
  // We also squash the b,t axes into s = block * time
  // CCGLIB requires that we have BLOCK x COMPLEX x COL x ROW structure.
  inline static const std::vector<int> modePlanarCons = {'c', 'p', 'z',
                                                         'f', 'n', 's'};
  inline static const std::vector<int> modePlanarColMajCons = {'c', 'p', 'z',
                                                               's', 'f', 'n'};

  inline static const std::vector<int> modeBeamCCGLIB{'c', 'p', 'z', 'e', 's'};
  inline static const std::vector<int> modeBeamOutput{'c', 'p', 'e', 's', 'z'};
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
      {'d', T::NR_PADDED_RECEIVERS},
      {'e', T::NR_BEAMS * 2}, // rfi mitigated beam + original beam
      {'f', T::NR_FPGA_SOURCES},
      {'g',
       T::NR_TIME_STEPS_PER_PACKET - 2 * NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE},
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

  BeamWeights *h_weights;
  // See the comment on this member in LambdaGPUPipeline (pipeline.hpp above)
  // -- same helper, same inert-when-unsteered contract.
  BeamSteering<T> beam_steering_;
  int *d_subpacket_delays;
  typename T::AntennaGains *d_gains;

  static constexpr int fft_total_packets_per_block =
      T::NR_CHANNELS * T::NR_PACKETS_FOR_CORRELATION * T::NR_FPGA_SOURCES;
  int fft_missing_packets;

public:
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {

    auto &b = buffers[current_buffer];
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

    LambdaPipelineIngest<T>::ingest_and_scale(
        this->state_, packet_data, b.stream, b.host_stream,
        b.samples_entry.get(), b.scales.get(), d_gains, b.samples_half.get(),
        dummy_run);

    tensor_16.runPermutation("packetToPreAlign", alpha,
                             (__half *)b.samples_half.get(),
                             (__half *)b.samples_pre_align.get(), b.stream);

    apply_delays_launch((__half *)b.samples_pre_align.get(),
                        (__half *)b.samples_aligned.get(), d_subpacket_delays,
                        T::NR_RECEIVERS_PER_PACKET, T::NR_FPGA_SOURCES,
                        T::NR_PACKETS_FOR_CORRELATION, T::NR_POLARIZATIONS,
                        T::NR_CHANNELS, T::NR_TIME_STEPS_PER_PACKET, b.stream);

    tensor_16.runPermutation("alignedToPlanar", alpha,
                             (__half *)b.samples_aligned.get(),
                             (__half *)b.samples_consolidated.get(), b.stream);

    tensor_16.runPermutation(
        "consToColMajCons", alpha, (__half *)b.samples_consolidated.get(),
        (__half *)b.samples_consolidated_col_maj.get(), b.stream);

    tensor_16.runPermutation(
        "alignedToPadding", alpha,
        reinterpret_cast<__half *>(b.samples_aligned.get()),
        reinterpret_cast<__half *>(b.samples_padding.get()), b.stream);

    // ------------------------------------------------------------------
    // 5. Copy unpadded → padded buffer then zero-fill the padding region
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpyAsync(b.samples_padded.get(), b.samples_padding.get(),
                               sizeof(typename T::HalfPacketAlignedSamplesType),
                               cudaMemcpyDefault, b.stream));
    CUDA_CHECK(
        cudaMemsetAsync(reinterpret_cast<char *>(b.samples_padded.get()) +
                            sizeof(typename T::HalfPacketAlignedSamplesType),
                        0,
                        sizeof(typename T::PaddedPacketSamplesType) -
                            sizeof(typename T::HalfPacketAlignedSamplesType),
                        b.stream));

    // ------------------------------------------------------------------
    // 6. Permute padded → correlator input layout
    // ------------------------------------------------------------------
    tensor_16.runPermutation(
        "paddedToCorrInput", alpha,
        reinterpret_cast<__half *>(b.samples_padded.get()),
        reinterpret_cast<__half *>(b.correlator_input.get()), b.stream);

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
        reinterpret_cast<void *>(b.decomp_visibilities.get()), T::NR_RECEIVERS,
        CUDA_R_32F, reinterpret_cast<void *>(b.eigenvalues.get()), CUDA_C_32F,
        b.cusolver_work_device.get(), b.cusolver_work_device_size,
        b.cusolver_work_host, b.cusolver_work_host_size, b.cusolver_info.get(),
        CUSOLVER_BATCH_SIZE));

    // ------------------------------------------------------------------
    // 11. Build projection filter and convert to fp16.
    //
    //  Null mode  (shrink_eigenvalues_=false):
    //    projection_matrix = I - U U^H, built entirely on-device from the
    //    last K columns of V for each batch.
    //
    //  Shrink mode (shrink_eigenvalues_=true):
    //    d[k] = 1               for k < N-K  (non-RFI, keep)
    //    d[k] = λ̄ / λ_k         for k ≥ N-K  (RFI, attenuate; λ̄=mean non-RFI)
    //    projection_matrix = V_scaled * V^H  (fp16)
    //    Scale factors are derived entirely on-device from the eigenvalues.
    // ------------------------------------------------------------------
    {
      constexpr int N = T::NR_RECEIVERS;
      const cuComplex gemm_alpha{1.0f, 0.0f};
      const cuComplex gemm_beta{0.0f, 0.0f};
      auto *V_base = reinterpret_cast<cuComplex *>(b.decomp_visibilities.get());
      auto *P_base =
          reinterpret_cast<cuComplex *>(b.float_projection_matrix.get());
      int32_t *counts_ptr = b.d_fixed_detected_counts.get();
      if (detect_signal_eigenmodes_) {
        detectSignalEigenmodeCounts(
            reinterpret_cast<const float *>(b.eigenvalues.get()),
            b.d_detected_counts.get(), N, CUSOLVER_BATCH_SIZE,
            detection_threshold_delta_, b.stream);
        counts_ptr = b.d_detected_counts.get();
      }

      if (!shrink_eigenvalues_) {
        buildIdentityMinusProjectionFromEigenvectors(
            reinterpret_cast<const float2 *>(b.decomp_visibilities.get()),
            counts_ptr, reinterpret_cast<__half2 *>(b.projection_matrix.get()),
            N, CUSOLVER_BATCH_SIZE, b.stream);
      } else {
        // --- Shrink path: M = V_scaled * V^H, projection_matrix = M (fp16) ---
        computeShrinkScaleFactors(
            reinterpret_cast<const float *>(b.eigenvalues.get()), counts_ptr,
            reinterpret_cast<float *>(b.d_scale_factors.get()), N,
            CUSOLVER_BATCH_SIZE, b.stream);

        scaleEigenvectorColumns(
            reinterpret_cast<const float2 *>(b.decomp_visibilities.get()),
            reinterpret_cast<float2 *>(b.scaled_eigenvectors.get()),
            reinterpret_cast<const float *>(b.d_scale_factors.get()), N,
            CUSOLVER_BATCH_SIZE, b.stream);

        auto *Vs_base =
            reinterpret_cast<cuComplex *>(b.scaled_eigenvectors.get());
        auto *M_base =
            reinterpret_cast<cuComplex *>(b.float_projection_matrix.get());

        for (int i = 0; i < CUSOLVER_BATCH_SIZE; ++i) {
          CUBLAS_CHECK(cublasGemmEx(
              b.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_C, N, N, N, &gemm_alpha,
              Vs_base + i * N * N, CUDA_C_32F, N, V_base + i * N * N,
              CUDA_C_32F, N, &gemm_beta, M_base + i * N * N, CUDA_C_32F, N,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }

        convert_float_to_half(
            reinterpret_cast<float *>(b.float_projection_matrix.get()),
            reinterpret_cast<__half *>(b.projection_matrix.get()),
            sizeof(FloatProjectionMatrix) / sizeof(float), b.stream);
      }
    }

    // conjugateMatrix((__half2 *)b.projection_matrix.get(),
    //                 T::NR_RECEIVERS * T::NR_RECEIVERS * T::NR_CHANNELS *
    //                     T::NR_POLARIZATIONS,
    //                 b.stream);

    // Slot 0 of weights_rfi_mitigated holds the original (un-projected) beam
    // weights. It was initialised once in the constructor, but
    // beam_steering_.maybe_refresh() above may have just written new steered
    // weights into b.weights -- leaving slot 0 frozen at the construction-time
    // (un-steered) direction. After steering fires, beam[0] points at the
    // initial uniform direction while beam[1] points at the steered+projected
    // target, making the RFI-mitigated beam appear to have MORE power than the
    // "original" beam. Update slot 0 from b.weights every frame so both beams
    // always use the same current steering.
    tensor_16.runPermutation("weightsToBeamMajor", alpha,
                             (__half *)b.weights.get(),
                             (__half *)b.weights_rfi_mitigated.get(), b.stream);

    {
      b.gemm_weight_projection_handle->Run(
          (CUdeviceptr)b.weights.get(), (CUdeviceptr)b.projection_matrix.get(),
          (CUdeviceptr)b.weights_updated.get());
    }

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

    b.gemm_handle->Run((CUdeviceptr)b.weights_beamformer.get(),
                       (CUdeviceptr)b.samples_consolidated_col_maj.get(),
                       (CUdeviceptr)b.beamformer_output.get());

    tensor_32.runPermutation("beamCCGLIBToBeamOutput", alpha_32,
                             (float *)b.beamformer_output.get(),
                             (float *)b.beam_output_float.get(), b.stream);

    convert_float_to_half((float *)b.beam_output_float.get(),
                          (__half *)b.beam_output.get(),
                          sizeof(BeamOutput) / sizeof(__half), b.stream);

    tensor_32.runPermutation("beamToCUFFTInput", alpha_32,
                             (float *)b.beamformer_output.get(),
                             (float *)b.samples_cufft_input.get(), b.stream);

    CUFFT_CHECK(cufftXtExec(b.fft_plan, (void *)b.samples_cufft_input.get(),
                            (void *)b.samples_cufft_output.get(),
                            CUFFT_FORWARD));

    tensor_32.runPermutation("cufftOutputToFineChannelRemove", alpha_32,
                             (float *)b.samples_cufft_output.get(),
                             (float *)b.samples_cufft_output_fine_channel.get(),
                             b.stream);

    // First half goes in second half of destination
    void *src_ptr = (char *)b.samples_cufft_output_fine_channel.get();
    size_t src_size = sizeof(FineChannelCopyType);

    dest_ptr = (char *)b.samples_fine_channel_removed.get() +
               sizeof(FineChannelCopyType);

    cudaMemcpyAsync(dest_ptr, src_ptr, src_size, cudaMemcpyDefault, b.stream);

    src_ptr = (char *)b.samples_cufft_output_fine_channel.get() +
              sizeof(FineChannelSeekType);
    dest_ptr = (char *)b.samples_fine_channel_removed.get();
    cudaMemcpyAsync(dest_ptr, src_ptr, src_size, cudaMemcpyDefault, b.stream);

    tensor_32.runPermutation("fineChannelRemovedToBeamFFTDownsample", alpha_32,
                             (float *)b.samples_fine_channel_removed.get(),
                             (float *)b.cufft_downsampled_input.get(),
                             b.stream);

    cudaMemsetAsync((float *)b.cufft_downsampled_output.get(), 0,
                    sizeof(FFTOutputType), b.stream);

    sum_fft_over_packets_launch(
        (float2 *)b.cufft_downsampled_input.get(),
        (float *)b.cufft_downsampled_output.get(), 2 * T::NR_BEAMS,
        T::NR_CHANNELS, T::NR_POLARIZATIONS,
        T::NR_TIME_STEPS_PER_PACKET - 2 * NR_FINE_CHANNELS_TO_REMOVE_EACH_SIDE,
        T::NR_PACKETS_FOR_CORRELATION, b.stream);

    if (output_ != nullptr && !dummy_run) {
      // -1, -1 is required but not used. Interface allows for single channel /
      // pol to be passed but this implementation does not use it.

      size_t beam_block_num =
          output_->register_beam_data_block(start_seq_num, end_seq_num);
      if (beam_block_num != std::numeric_limits<size_t>::max()) {
        auto *beam_output_pointer =
            (void *)output_->get_beam_data_landing_pointer(beam_block_num);

        cudaMemcpyAsync(beam_output_pointer, b.beam_output.get(),
                        sizeof(BeamOutput), cudaMemcpyDefault, b.stream);

        auto *beam_output_ctx = new OutputTransferCompleteContext{
            .output = this->output_, .block_index = beam_block_num};

        cudaLaunchHostFunc(b.stream, output_transfer_complete_host_func,
                           beam_output_ctx);

        bool *arrivals_output_pointer =
            (bool *)output_->get_arrivals_data_landing_pointer(beam_block_num);
        std::memcpy(arrivals_output_pointer, packet_data->get_arrivals_ptr(),
                    packet_data->get_arrivals_size());
        output_->register_arrivals_transfer_complete(beam_block_num);
      }

      size_t fft_block_num =
          output_->register_fft_block(start_seq_num, end_seq_num);
      if (fft_block_num != std::numeric_limits<size_t>::max()) {
        auto *fft_output_pointer =
            (void *)output_->get_fft_landing_pointer(fft_block_num);
        cudaMemcpyAsync(fft_output_pointer, b.cufft_downsampled_output.get(),
                        sizeof(FFTOutputType), cudaMemcpyDefault, b.stream);

        auto *fft_output_ctx = new OutputTransferCompleteContext{
            .output = this->output_, .block_index = fft_block_num};
        cudaLaunchHostFunc(b.stream, fft_output_transfer_complete_host_func,
                           fft_output_ctx);
      }

      size_t eig_block_num = output_->register_eigendecomposition_data_block(
          start_seq_num, end_seq_num);

      if (eig_block_num != std::numeric_limits<size_t>::max()) {
        void *eigval_ptr =
            output_->get_eigenvalues_data_landing_pointer(eig_block_num);
        void *eigvec_ptr =
            output_->get_eigenvectors_data_landing_pointer(eig_block_num);
        void *eigcount_ptr =
            output_->get_nulled_eigenmode_counts_landing_pointer(eig_block_num);
        CUDA_CHECK(cudaMemcpyAsync(eigvec_ptr, b.decomp_visibilities.get(),
                                   sizeof(DecompositionVisibilities),
                                   cudaMemcpyDefault, b.stream));

        CUDA_CHECK(cudaMemcpyAsync(eigval_ptr, b.eigenvalues.get(),
                                   sizeof(Eigenvalues), cudaMemcpyDefault,
                                   b.stream));
        if (eigcount_ptr != nullptr) {
          const int32_t *counts_ptr = detect_signal_eigenmodes_
                                          ? b.d_detected_counts.get()
                                          : b.d_fixed_detected_counts.get();
          CUDA_CHECK(cudaMemcpyAsync(
              b.h_detected_counts_staging, counts_ptr,
              sizeof(int32_t) * CUSOLVER_BATCH_SIZE, cudaMemcpyDeviceToHost,
              b.stream));
        }

        auto *ctx = new EigenOutputTransferWithCountsContext{
            .output = this->output_,
            .block_index = eig_block_num,
            .counts_dst = eigcount_ptr,
            .counts_src = b.h_detected_counts_staging,
            .counts_size_bytes = sizeof(int32_t) * CUSOLVER_BATCH_SIZE,
            .stats_mutex = b.eigenmode_stats_mutex.get(),
            .stats_history = b.eigenmode_stats_history.get()};
        CUDA_CHECK(cudaLaunchHostFunc(
            b.stream, eigen_output_transfer_with_counts_host_func, ctx));
      }
    }

    // Rotate buffer indices
    if (!dummy_run) {
      current_buffer = (current_buffer + 1) % num_buffers;
    }
  }

  void print_eigenmode_stats() {
    std::vector<int32_t> samples;
    for (auto &buffer : buffers) {
      if (!buffer.eigenmode_stats_mutex || !buffer.eigenmode_stats_history) {
        continue;
      }
      std::lock_guard<std::mutex> lock(*buffer.eigenmode_stats_mutex);
      if (buffer.eigenmode_stats_history->empty()) {
        continue;
      }
      samples.insert(samples.end(), buffer.eigenmode_stats_history->begin(),
                     buffer.eigenmode_stats_history->end());
      buffer.eigenmode_stats_history->clear();
    }

    const size_t per_sample = CUSOLVER_BATCH_SIZE;
    if (samples.size() % per_sample != 0) {
      return;
    }
    const size_t sample_count = samples.size() / per_sample;
    if (sample_count == 0) {
      return;
    }

    std::cout << "Eigenmode nulling stats over last "
              << sample_count << " samples" << std::endl;
    for (int channel = 0; channel < T::NR_CHANNELS; ++channel) {
      for (int pol = 0; pol < T::NR_POLARIZATIONS; ++pol) {
        const size_t flat = channel * T::NR_POLARIZATIONS + pol;
        std::vector<int32_t> series;
        series.reserve(sample_count);
        for (size_t sample = 0; sample < sample_count; ++sample) {
          series.push_back(samples[sample * per_sample + flat]);
        }
        std::sort(series.begin(), series.end());
        const int32_t min_count = series.front();
        const int32_t max_count = series.back();
        double sum = 0.0;
        for (int32_t count : series) {
          sum += static_cast<double>(count);
        }
        const double mean = sum / static_cast<double>(series.size());
        const double median =
            (series.size() % 2 == 0)
                ? 0.5 * (static_cast<double>(series[series.size() / 2 - 1]) +
                         static_cast<double>(series[series.size() / 2]))
                : static_cast<double>(series[series.size() / 2]);

        std::cout << "  ch=" << channel << " pol=" << pol
                  << " min=" << min_count << " max=" << max_count
                  << " median=" << median << " mean=" << mean << std::endl;
      }
    }
    std::fflush(stdout);
  }
  LambdaAdaptiveBeamformedSpectraPipeline(
      const int num_buffers, BeamWeightsT<T> *h_weights,
      const std::unordered_map<int, int> nr_signal_eigenvectors,
      const int min_freq_channel, BeamSteering<T> beam_steering,
      bool shrink_eigenvalues = false, bool detect_signal_eigenmodes = false,
      float detection_threshold_delta = 3.0f,
      double eigenmode_stats_interval_seconds = 10.0)

      : num_buffers(num_buffers), h_weights(h_weights),
        beam_steering_(std::move(beam_steering)),
        correlator(cu::Device(0), tcc::Format::fp16, T::NR_PADDED_RECEIVERS,
                   T::NR_CHANNELS,
                   NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK,
                   T::NR_POLARIZATIONS, std::nullopt,
                   T::NR_PADDED_RECEIVERS_PER_BLOCK, TCC_THREAD_BLOCKS_PER_SM),
        tensor_16(extent, CUTENSOR_R_16F, 128),
        tensor_32(extent, CUTENSOR_R_32F, 128),
        NR_SIGNAL_EIGENVECTORS(nr_signal_eigenvectors),
        min_freq_channel(min_freq_channel),
        shrink_eigenvalues_(shrink_eigenvalues),
        detect_signal_eigenmodes_(detect_signal_eigenmodes),
        detection_threshold_delta_(detection_threshold_delta),
        eigenmode_stats_interval_seconds_(eigenmode_stats_interval_seconds) {
    std::cout << "Beamformed Spectra instantiated with NR_CHANNELS: "
              << T::NR_CHANNELS << ", NR_RECEIVERS: " << T::NR_RECEIVERS
              << ", NR_POLARIZATIONS: " << T::NR_POLARIZATIONS
              << ", NR_SAMPLES_PER_CHANNEL: "
              << NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK
              << ", NR_TIMES_PER_BLOCK: " << NR_TIMES_PER_BLOCK
              << ", NR_BLOCKS_FOR_FFT: " << NR_BLOCKS_FOR_CORRELATION
              << ", NR_BEAMS: " << 2 * T::NR_BEAMS << std::endl;

    const long long CUFFT_FFT_SIZE = 64;
    long long N[] = {CUFFT_FFT_SIZE};
    const size_t NUM_TOTAL_BATCHES = 2 * T::NR_BEAMS * T::NR_CHANNELS *
                                     T::NR_POLARIZATIONS *
                                     T::NR_PACKETS_FOR_CORRELATION;

    size_t work_size = 0;
    {
      // Temporary plan to calculate work_size
      cufftHandle temp_plan;
      CUFFT_CHECK(cufftCreate(&temp_plan));
      CUFFT_CHECK(cufftXtMakePlanMany(temp_plan, 1, N, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NUM_TOTAL_BATCHES, &work_size,
                                      CUDA_C_32F));
      cufftDestroy(temp_plan);
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

    tensor_32.addTensor(modeCUFFTInput, "cufftInput");
    tensor_32.addTensor(modeBeamCCGLIB, "beamCCGLIB");
    tensor_32.addTensor(modeBeamOutput, "beamOutput");

    tensor_16.addTensor(modeCorrelatorInput, "corr_input");
    tensor_32.addTensor(modeVisCorr, "visCorr");
    tensor_32.addTensor(modeVisCorrBaseline, "visBaseline");
    tensor_32.addTensor(modeVisCorrBaselineTrimmed, "visBaselineTrimmed");
    tensor_32.addTensor(modeVisDecomp, "visDecomp");
    tensor_32.addTensor(modeCUFFTOutput, "cufftOutput");
    tensor_32.addTensor(modeFineChannelRemove, "fineChannelRemove");
    tensor_32.addTensor(modeFineChannelRemoved, "fineChannelRemoved");
    tensor_32.addTensor(modeBeamFFTDownsample, "beamFFTDownsample");

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
    tensor_32.addPermutation("beamCCGLIB", "cufftInput",
                             CUTENSOR_COMPUTE_DESC_32F, "beamToCUFFTInput");
    tensor_32.addPermutation("cufftOutput", "fineChannelRemove",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "cufftOutputToFineChannelRemove");

    tensor_32.addPermutation("beamCCGLIB", "beamOutput",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "beamCCGLIBToBeamOutput");
    tensor_32.addPermutation("fineChannelRemoved", "beamFFTDownsample",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "fineChannelRemovedToBeamFFTDownsample");

    CUDA_CHECK(cudaMalloc((void **)&d_subpacket_delays,
                          sizeof(int) * T::NR_FPGA_SOURCES));
    CUDA_CHECK(cudaMalloc((void **)&d_gains, sizeof(typename T::AntennaGains)));

    CUDA_CHECK(
        cudaMemset(d_subpacket_delays, 0, sizeof(int) * T::NR_FPGA_SOURCES));

    auto default_gains = get_default_gains<T::NR_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));

    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);
    buffers.reserve(num_buffers);
    std::vector<int32_t> fixed_detected_counts(CUSOLVER_BATCH_SIZE, 0);
    for (int channel = 0; channel < T::NR_CHANNELS; ++channel) {
      const auto it = NR_SIGNAL_EIGENVECTORS.find(min_freq_channel + channel);
      const int fixed_k =
          (it != NR_SIGNAL_EIGENVECTORS.end()) ? it->second : 0;
      for (int pol = 0; pol < T::NR_POLARIZATIONS; ++pol) {
        fixed_detected_counts[channel * T::NR_POLARIZATIONS + pol] = fixed_k;
      }
    }
    for (int i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(cu_device, work_size);

      // Finalize cuFFT plan for this buffer
      auto &b = buffers.back();
      CUFFT_CHECK(cufftXtMakePlanMany(b.fft_plan, 1, N, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NUM_TOTAL_BATCHES, &work_size,
                                      CUDA_C_32F));
      CUFFT_CHECK(cufftSetStream(b.fft_plan, b.stream));
      CUFFT_CHECK(cufftSetWorkArea(b.fft_plan, b.cufft_work_area.get()));

      // Copy initial weights
      cudaMemcpyAsync(b.weights.get(), h_weights, sizeof(BeamWeights),
                      cudaMemcpyDefault, b.stream);
      tensor_16.runPermutation("weightsToBeamMajor", alpha,
                               (__half *)b.weights.get(),
                               (__half *)b.weights_permuted.get(), b.stream);
      cudaMemcpyAsync(b.weights_rfi_mitigated.get(), b.weights_permuted.get(),
                      sizeof(BeamWeights), cudaMemcpyDefault, b.stream);
      CUDA_CHECK(cudaMemcpyAsync(
          b.d_fixed_detected_counts.get(), fixed_detected_counts.data(),
          sizeof(int32_t) * CUSOLVER_BATCH_SIZE, cudaMemcpyHostToDevice,
          b.stream));
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
  };

  void dump_visibilities(const uint64_t end_seq_num = 0) override {
    // nothing to do.
  }

  virtual void set_subpacket_delays(int *delays_subpacket) {
    subpacket_delays_ = delays_subpacket;
    CUDA_CHECK(cudaMemcpy(d_subpacket_delays, subpacket_delays_,
                          sizeof(int) * T::NR_FPGA_SOURCES, cudaMemcpyDefault));
  }
};
