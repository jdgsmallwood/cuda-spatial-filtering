#pragma once

template <typename T, bool RFI_MITIGATE = false>
class LambdaPulsarFoldPipeline : public GPUPipeline {
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
                                [NR_TIME_STEPS_FOR_CORRELATION][COMPLEX];
  using BeamOutput = float[num_beams][NR_TIME_STEPS_FOR_CORRELATION]
                          [T::NR_CHANNELS][T::NR_POLARIZATIONS][COMPLEX];

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
        samples_consolidated, samples_consolidated_col_maj, samples_padding;
    DevicePtr<typename T::PaddedPacketSamplesType> samples_padded;
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
          samples_consolidated(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_consolidated_col_maj(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_padding(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          beam_output(make_device_ptr<BeamOutput>()),
          weights(make_device_ptr<BeamWeights>()),
          weights_permuted(make_device_ptr<BeamWeights>()),
          weights_updated(make_device_ptr<BeamWeights>()),
          weights_rfi_mitigated(make_device_ptr<RFIMitigatedBeamWeights>()),
          weights_beamformer(make_device_ptr<ChosenBeamWeights>()),
          beamformer_output(make_device_ptr<BeamformerOutput>()),
          samples_padded(
              make_device_ptr<typename T::PaddedPacketSamplesType>()),
          correlator_input(make_device_ptr<CorrelatorInput>()),
          correlator_output(make_device_ptr<CorrelatorOutput>()),
          visibilities_baseline(make_device_ptr<Visibilities>()),
          visibilities_trimmed_baseline(make_device_ptr<TrimmedVisibilities>()),
          visibilities_trimmed(make_device_ptr<TrimmedVisibilities>()),
          decomp_visibilities(make_device_ptr<DecompositionVisibilities>()),
          projection_matrix(make_device_ptr<ProjectionMatrix>()),
          float_projection_matrix(make_device_ptr<FloatProjectionMatrix>()),
          eigenvalues(make_device_ptr<Eigenvalues>()),
          cusolver_info(
              make_device_ptr<int>(CUSOLVER_BATCH_SIZE * sizeof(int))) {
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
      // if RFI mitigate - we will have double the number of beams.
      gemm_handle = std::make_unique<ccglib::pipeline::Pipeline>(
          T::NR_CHANNELS * T::NR_POLARIZATIONS, num_beams,
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

  inline static const std::vector<int> modePacket{'c', 'y', 'f', 'u',
                                                  'n', 'p', 'z'};
  inline static const std::vector<int> modePlanar{'c', 'p', 'z', 'f',
                                                  'n', 'o', 'u'};

  inline static const std::vector<int> modePacketPreAlign{'f', 'y', 'u', 'c',
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
  char *d_obs_header;

  BeamWeights *h_weights;
  // See the comment on this member in LambdaGPUPipeline (pipeline.hpp above)
  // -- same helper, same inert-when-unsteered contract.
  BeamSteering<T> beam_steering_;
  int *d_subpacket_delays;
  typename T::AntennaGains *d_gains;

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

    if (RFI_MITIGATE) {
      tensor_16.runPermutation(
          "alignedToPadding", alpha,
          reinterpret_cast<__half *>(b.samples_aligned.get()),
          reinterpret_cast<__half *>(b.samples_padding.get()), b.stream);

      // ------------------------------------------------------------------
      // 5. Copy unpadded → padded buffer then zero-fill the padding region
      // ------------------------------------------------------------------
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

    if (output_ != nullptr && !dummy_run) {

      // PSRDADA beam streaming -- the header/data block writes below all
      // dereference `hdu`, which is null when the sink is disabled
      // (dada_key == 0). Skip them in that case; the eigen-output block that
      // follows uses `output_` (not PSRDADA) and is intentionally left
      // outside this guard.
      if (dada_key != 0) {
        if (!header_written) {
          std::cout << "writing header...\n";
          uint64_t rfi_header_size = 0;
          uint64_t header_size = ipcbuf_get_bufsz(hdu->header_block);
          char *header = ipcbuf_get_next_write(hdu->header_block);
          cudaMemcpyAsync(header, d_obs_header, header_size, cudaMemcpyDefault,
                          b.stream);

          if constexpr (RFI_MITIGATE) {

            rfi_header_size = ipcbuf_get_bufsz(rfi_hdu->header_block);
            char *rfi_header = ipcbuf_get_next_write(rfi_hdu->header_block);

            cudaMemcpyAsync(rfi_header, d_obs_header, header_size,
                            cudaMemcpyDefault, b.stream);
          }

          // // Enable EOD so that subsequent transfers will move to the next
          // buffer
          // // in the header block
          // if (ipcbuf_enable_eod(hdu->header_block) < 0) {
          //   multilog(log, LOG_ERR, "Could not enable EOD on Header Block\n");
          // }

          cudaDeviceSynchronize();
          // flag the header block for this "observation" as filled
          if (ipcbuf_mark_filled(hdu->header_block, header_size) < 0) {
            multilog(log, LOG_ERR, "could not mark filled Header Block\n");
            std::cout << "could not mark filled header block...\n";
          }

          if constexpr (RFI_MITIGATE) {
            if (ipcbuf_mark_filled(rfi_hdu->header_block, rfi_header_size) <
                0) {
              multilog(log, LOG_ERR, "could not mark filled Header Block\n");
              std::cout << "could not mark filled header block...\n";
            }
          }
          header_written = true;
        }

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
          cudaMemcpyAsync(block, (char *)b.beam_output.get(), block_size,
                          cudaMemcpyDefault, b.stream);

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
            cudaMemcpyAsync(rfi_block, (char *)b.beam_output.get() + block_size,
                            rfi_block_size, cudaMemcpyDefault, b.stream);
          }

          cudaDeviceSynchronize();

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

    // Rotate buffer indices
  }
  LambdaPulsarFoldPipeline(
      BeamWeightsT<T> *h_weights,
      const std::unordered_map<int, int> nr_signal_eigenvectors,
      const int min_freq_channel, key_t dada_key, std::string header_filename,
      key_t rfi_dada_key, BeamSteering<T> beam_steering)

      : num_buffers(1), h_weights(h_weights),
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
    d_obs_header = nullptr;
    if (dada_key != 0) {
      log = multilog_open("pulsar_fold_writer", 0);
      multilog_add(log, stderr);
      hdu = dada_hdu_create(log);
      dada_hdu_set_key(hdu, dada_key);
      // connect to HDU
      if (dada_hdu_connect(hdu) < 0) {
        multilog(log, LOG_ERR, "could not connect to HDU\n");
      }

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
        fprintf(stderr, "ERROR: could not read ASCII header from %s\n",
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

      cudaMalloc(&d_obs_header, DADA_DEFAULT_HEADER_SIZE);
      cudaMemcpy(d_obs_header, obs_header, DADA_DEFAULT_HEADER_SIZE,
                 cudaMemcpyDefault);
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

    auto default_gains = get_default_gains<T::NR_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));

    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);
    buffers.reserve(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(cu_device);

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
  };

  ~LambdaPulsarFoldPipeline() {
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
    // nothing to do.
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
    return NR_TIME_STEPS_FOR_CORRELATION;
  }
};
