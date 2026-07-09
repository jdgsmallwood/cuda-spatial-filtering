#pragma once

template <typename T> class LambdaGPUPipeline : public GPUPipeline {

private:
  int num_buffers;

  // We are converting it to fp16 so this should not be changable anymore.
  static constexpr int NR_TIMES_PER_BLOCK = 128 / 16; // NR_BITS;

  static constexpr int NR_BLOCKS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET /
      NR_TIMES_PER_BLOCK;
  static constexpr int NR_BASELINES =
      T::NR_PADDED_RECEIVERS * (T::NR_PADDED_RECEIVERS + 1) / 2;
  static constexpr int NR_UNPADDED_BASELINES =
      T::NR_RECEIVERS * (T::NR_RECEIVERS + 1) / 2;
  static constexpr int NR_TIME_STEPS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET;
  static constexpr int COMPLEX = 2;
  static constexpr int NR_EIGENVALUES =
      T::NR_PADDED_RECEIVERS * T::NR_CHANNELS * T::NR_POLARIZATIONS;
  const int NR_CORRELATED_BLOCKS_TO_ACCUMULATE;

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

  using TrimmedVisibilities =
      std::complex<float>[T::NR_CHANNELS][NR_UNPADDED_BASELINES]
                         [T::NR_POLARIZATIONS][T::NR_POLARIZATIONS];

  using DecompositionVisibilities =
      std::complex<float>[T::NR_CHANNELS][T::NR_POLARIZATIONS]
                         [T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                         [T::NR_RECEIVERS];
  using Eigenvalues = float[T::NR_CHANNELS][T::NR_POLARIZATIONS]
                           [T::NR_POLARIZATIONS][T::NR_RECEIVERS];

  using BeamformerInput =
      __half[T::NR_CHANNELS][NR_BLOCKS_FOR_CORRELATION][T::NR_PADDED_RECEIVERS]
            [T::NR_POLARIZATIONS][NR_TIMES_PER_BLOCK][COMPLEX];

  using BeamformerOutput =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
           [NR_TIME_STEPS_FOR_CORRELATION][COMPLEX];

  using HalfBeamformerOutput =
      __half[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
            [NR_TIME_STEPS_FOR_CORRELATION][COMPLEX];

  using BeamWeights = BeamWeightsT<T>;

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

  inline static const std::vector<int> modePacket{'c', 'g', 'f', 'u',
                                                  'n', 'p', 'z'};
  inline static const std::vector<int> modePacketPreAlign{'f', 'g', 'u', 'c',
                                                          'n', 'p', 'z'};
  inline static const std::vector<int> modePacketAligned{'f', 'o', 'u', 'c',
                                                         'n', 'p', 'z'};
  // o and u need to end up together and will be interpreted as b x t in the
  // next transformation. Similarly f x n = r in next transformation.
  inline static const std::vector<int> modePacketPadding{'f', 'n', 'c', 'o',
                                                         'u', 'p', 'z'};
  inline static const std::vector<int> modePacketPadded{'d', 'c', 'b',
                                                        't', 'p', 'z'};
  inline static const std::vector<int> modeCorrelatorInput{'c', 'b', 'd',
                                                           'p', 't', 'z'};
  inline static const std::vector<int> modePlanar{'c', 'p', 'z', 'f',
                                                  'n', 'o', 'u'};
  inline static const std::vector<int> modeCUFFTInput{'c', 'p', 'm', 's', 'z'};
  // We need the planar samples matrix to be in column-major memory layout
  // which is equivalent to transposing time and receiver structure here.
  // We also squash the b,t axes into s = block * time
  // CCGLIB requires that we have BLOCK x COMPLEX x COL x ROW structure.
  inline static const std::vector<int> modePlanarCons = {'c', 'p', 'z',
                                                         'f', 'n', 's'};
  inline static const std::vector<int> modePlanarColMajCons = {'c', 'p', 'z',
                                                               's', 'f', 'n'};
  // Flattened view of samples_aligned: o*u merged into s for direct fusion
  // to planarColMajCons layout (alignedToColMajCons permutation).
  inline static const std::vector<int> modePacketAlignedFlat{'f', 's', 'c',
                                                              'n', 'p', 'z'};
  inline static const std::vector<int> modeVisCorr{'c', 'l', 'p', 'q', 'z'};
  inline static const std::vector<int> modeVisCorrBaseline{'l', 'c', 'p', 'q',
                                                           'z'};
  inline static const std::vector<int> modeVisCorrBaselineTrimmed{'a', 'c', 'p',
                                                                  'q', 'z'};

  inline static const std::vector<int> modeVisCorrTrimmed{'c', 'a', 'p', 'q',
                                                          'z'};
  inline static const std::vector<int> modeVisDecomp{'c', 'p', 'q', 'a', 'z'};
  // Convert back to interleaved instead of planar output.
  // This is not strictly necessary to do in the pipeline.
  inline static const std::vector<int> modeBeamCCGLIB{'c', 'p', 'z', 'm', 's'};
  inline static const std::vector<int> modeBeamOutput{'c', 'p', 'm', 's', 'z'};
  inline static const std::vector<int> modeWeightsInput{'c', 'p', 'm', 'r',
                                                        'z'};
  inline static const std::vector<int> modeWeightsCCGLIB{'c', 'p', 'z', 'm',
                                                         'r'};

  inline static const std::unordered_map<int, int64_t> extent = {

      {'a', NR_UNPADDED_BASELINES},
      {'b', NR_BLOCKS_FOR_CORRELATION},
      {'c', T::NR_CHANNELS},
      {'d', T::NR_PADDED_RECEIVERS},
      {'f', T::NR_FPGA_SOURCES},
      {'g', T::NR_PACKETS_FOR_CORRELATION + 2},
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
      {'z', 2}, // real, imaginary

  };

  CutensorSetup tensor_16;
  CutensorSetup tensor_32;

  int current_buffer;
  std::atomic<int> last_frame_processed;
  std::atomic<int> num_integrated_units_processed;
  std::atomic<int> num_correlation_units_integrated;

  tcc::Correlator correlator;

  struct PipelineResources {
    cudaStream_t stream;
    cudaStream_t host_stream;
    ManagedCufftPlan fft_plan;

    DevicePtr<typename T::InputPacketSamplesType> samples_entry; // y
    DevicePtr<typename T::PacketScalesType> scales;              // y
    DevicePtr<typename T::HalfPacketSamplesType> samples_half,
        samples_pre_align; // y
    DevicePtr<typename T::HalfPacketAlignedSamplesType> samples_aligned,
        samples_padding, samples_consolidated,
        samples_consolidated_col_maj;                                      // y
    DevicePtr<typename T::PaddedPacketSamplesType> samples_padded;         // y
    DevicePtr<typename T::FFTCUFFTInputType> samples_cufft_input;          // y
    DevicePtr<typename T::FFTCUFFTOutputType> samples_cufft_output;        // y
    DevicePtr<typename T::FFTOutputType> cufft_downsampled_output;         // y
    DevicePtr<BeamWeights> weights, weights_permuted, weights_updated;     // y
    DevicePtr<BeamformerOutput> beamformer_output, beamformer_data_output; // y
    DevicePtr<HalfBeamformerOutput> beamformer_data_output_half;           // y
    DevicePtr<void> cufft_work_area;                                       // y

    // Correlator I/O
    DevicePtr<CorrelatorInput> correlator_input;   // y
    DevicePtr<CorrelatorOutput> correlator_output; // y

    // Intermediate visibility layout buffers
    DevicePtr<Visibilities> visibilities_baseline; // y
    DevicePtr<TrimmedVisibilities> visibilities_trimmed_baseline,
        visibilities_trimmed, visibilities_permuted; // y

    // Per-block correlation matrix (input to cuSOLVER, overwritten in-place
    // with eigenvectors on output).
    DevicePtr<DecompositionVisibilities> decomp_visibilities; // y

    // Per-block eigenvalues (ascending order from cuSOLVER).
    DevicePtr<Eigenvalues> eigenvalues; // y

    // cuSOLVER handles / workspace
    cusolverDnHandle_t cusolver_handle = nullptr;
    cusolverDnParams_t cusolver_params = nullptr;
    DevicePtr<int> cusolver_info; // [CUSOLVER_BATCH_SIZE]
    DevicePtr<void> cusolver_work_device;
    void *cusolver_work_host = nullptr;
    size_t cusolver_work_device_size = 0;
    size_t cusolver_work_host_size = 0;
    std::unique_ptr<ccglib::pipeline::Pipeline> gemm_handle;

    // Instantiated CUDA graphs of the two static mid-pipeline sections
    // (pre- and post-eigendecomposition).  All device pointers in those
    // sections are fixed per PipelineResources, so the capture stays valid
    // for the buffer's lifetime.  nullptr -> run the section eagerly
    // (capture failed or disabled via SPATIAL_DISABLE_CUDA_GRAPH).
    cudaGraphExec_t graph_pre = nullptr;
    cudaGraphExec_t graph_post = nullptr;

    // Recorded on `stream` right after the post-eigendecomposition section
    // (which includes the accumulate_visibilities add into
    // d_visibilities_accumulator) on every execute_pipeline call.
    // dump_visibilities waits on this from all buffers before reading the
    // accumulator, instead of a full cudaDeviceSynchronize().
    cudaEvent_t accumulate_done = nullptr;

    PipelineResources(CUdevice cu_device, size_t work_size)
        : samples_entry(make_device_ptr<typename T::InputPacketSamplesType>()),
          scales(make_device_ptr<typename T::PacketScalesType>()),
          samples_half(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_pre_align(
              make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_aligned(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_padding(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_consolidated(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_consolidated_col_maj(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_cufft_input(make_device_ptr<typename T::FFTCUFFTInputType>()),
          samples_cufft_output(
              make_device_ptr<typename T::FFTCUFFTOutputType>()),
          cufft_downsampled_output(
              make_device_ptr<typename T::FFTOutputType>()),
          beamformer_data_output_half(make_device_ptr<HalfBeamformerOutput>()),
          weights(make_device_ptr<BeamWeights>()),
          weights_permuted(make_device_ptr<BeamWeights>()),
          weights_updated(make_device_ptr<BeamWeights>()),
          beamformer_output(make_device_ptr<BeamformerOutput>()),
          beamformer_data_output(make_device_ptr<BeamformerOutput>()),
          samples_padded(
              make_device_ptr<typename T::PaddedPacketSamplesType>()),
          correlator_input(make_device_ptr<CorrelatorInput>()),
          correlator_output(make_device_ptr<CorrelatorOutput>()),
          visibilities_baseline(make_device_ptr<Visibilities>()),
          visibilities_trimmed_baseline(make_device_ptr<TrimmedVisibilities>()),
          visibilities_trimmed(make_device_ptr<TrimmedVisibilities>()),
          visibilities_permuted(make_device_ptr<TrimmedVisibilities>()),
          decomp_visibilities(make_device_ptr<DecompositionVisibilities>()),
          eigenvalues(make_device_ptr<Eigenvalues>()),
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
          T::NR_CHANNELS * T::NR_POLARIZATIONS, T::NR_BEAMS,
          NR_TIMES_PER_BLOCK * NR_BLOCKS_FOR_CORRELATION, T::NR_RECEIVERS,
          cu_device, stream, ccglib::complex_planar, ccglib::complex_planar,
          ccglib::mma::row_major, ccglib::mma::col_major,
          ccglib::mma::row_major, ccglib::ValueType::float16,
          ccglib::ValueType::float32, ccglib::mma::opt, alpha_ccglib,
          beta_ccglib);
    }

    ~PipelineResources() {
      if (graph_pre)
        cudaGraphExecDestroy(graph_pre);
      if (graph_post)
        cudaGraphExecDestroy(graph_post);
      if (accumulate_done)
        cudaEventDestroy(accumulate_done);
      cudaStreamDestroy(stream);
      cudaStreamDestroy(host_stream);
      if (cusolver_handle)
        cusolverDnDestroy(cusolver_handle);
      if (cusolver_params)
        cusolverDnDestroyParams(cusolver_params);
      if (cusolver_work_host)
        std::free(cusolver_work_host);
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
          samples_cufft_input(std::move(other.samples_cufft_input)),
          samples_cufft_output(std::move(other.samples_cufft_output)),
          cufft_downsampled_output(std::move(other.cufft_downsampled_output)),
          weights(std::move(other.weights)),
          weights_permuted(std::move(other.weights_permuted)),
          beamformer_output(std::move(other.beamformer_output)),
          cufft_work_area(std::move(other.cufft_work_area)),
          gemm_handle(std::move(other.gemm_handle)), graph_pre(other.graph_pre),
          graph_post(other.graph_post), accumulate_done(other.accumulate_done) {
      other.stream = nullptr;
      other.host_stream = nullptr;
      other.graph_pre = nullptr;
      other.graph_post = nullptr;
      other.accumulate_done = nullptr;
    }

    PipelineResources &operator=(PipelineResources &&other) noexcept {
      if (this != &other) {
        if (stream)
          cudaStreamDestroy(stream);
        if (host_stream)
          cudaStreamDestroy(host_stream);
        if (graph_pre)
          cudaGraphExecDestroy(graph_pre);
        if (graph_post)
          cudaGraphExecDestroy(graph_post);
        if (accumulate_done)
          cudaEventDestroy(accumulate_done);
        graph_pre = other.graph_pre;
        graph_post = other.graph_post;
        accumulate_done = other.accumulate_done;
        other.graph_pre = nullptr;
        other.graph_post = nullptr;
        other.accumulate_done = nullptr;

        stream = other.stream;
        host_stream = other.host_stream;
        fft_plan = std::move(other.fft_plan);
        samples_entry = std::move(other.samples_entry);
        scales = std::move(other.scales);
        samples_half = std::move(other.samples_half);
        samples_consolidated = std::move(other.samples_consolidated);
        samples_consolidated_col_maj =
            std::move(other.samples_consolidated_col_maj);
        samples_cufft_input = std::move(other.samples_cufft_input);
        samples_cufft_output = std::move(other.samples_cufft_output);
        cufft_downsampled_output = std::move(other.cufft_downsampled_output);
        weights = std::move(other.weights);
        weights_permuted = std::move(other.weights_permuted);
        beamformer_output = std::move(other.beamformer_output);
        cufft_work_area = std::move(other.cufft_work_area);
        gemm_handle = std::move(other.gemm_handle);

        other.stream = nullptr;
        other.host_stream = nullptr;
      }
      return *this;
    }

    // 3. Explicitly Delete Copying
    PipelineResources(const PipelineResources &) = delete;
    PipelineResources &operator=(const PipelineResources &) = delete;
  };

  BeamWeights *h_weights;
  // Periodically refreshes b.weights to track this pipeline's beam targets
  // (see compute_steering_weights()/BeamSteering above pipeline.hpp:46+ for
  // why and how) -- inert (a permanent no-op) when no --targets-filename was
  // supplied, in which case h_weights above remains the sole source of truth,
  // exactly as before this feature existed.
  BeamSteering<T> beam_steering_;
  TrimmedVisibilities *d_visibilities_accumulator;

  // Recorded on buffers[0].stream after dump_visibilities resets
  // d_visibilities_accumulator. Each buffer's post-eigen section waits on
  // this before its next accumulate_visibilities, so the reset can never
  // race with an in-flight accumulation -- without a cudaDeviceSynchronize().
  cudaEvent_t visibilities_reset_done = nullptr;

  typename T::AntennaGains *d_gains;
  std::vector<PipelineResources> buffers;
  int *d_subpacket_delays;
  int visibilities_start_seq_num;
  int visibilities_end_seq_num;
  static constexpr int visibilities_total_packets_per_block =
      T::NR_CHANNELS * T::NR_PACKETS_FOR_CORRELATION * T::NR_FPGA_SOURCES;
  int visibilities_missing_packets;
  cusolverEigMode_t cusolver_jobz;
  cublasFillMode_t cusolver_uplo;
  static constexpr int CUSOLVER_BATCH_SIZE =
      T::NR_CHANNELS * T::NR_POLARIZATIONS * T::NR_POLARIZATIONS;

public:
  static constexpr size_t NR_BENCHMARKING_RUNS = 100;
  size_t benchmark_runs_done = 0;
  cudaEvent_t start_run[NR_BENCHMARKING_RUNS], stop_run[NR_BENCHMARKING_RUNS];
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

    const size_t start_seq_num = packet_data->start_seq_id;
    const size_t end_seq_num = packet_data->end_seq_id;
    if (visibilities_start_seq_num == -1) {
      visibilities_start_seq_num = packet_data->start_seq_id;
    }
    visibilities_missing_packets += packet_data->get_num_missing_packets();

    // Record GPU start event
    cudaEventRecord(start_run[benchmark_runs_done], b.stream);

    LambdaPipelineIngest<T>::ingest_and_scale(
        this->state_, packet_data, b.stream, b.host_stream,
        b.samples_entry.get(), b.scales.get(), d_gains, b.samples_half.get(),
        dummy_run);

    // Pre-eigendecomposition section: a single graph launch replaces ~12
    // individual launches when capture succeeded at construction (see
    // enqueue_pre_eigen / capture_graph).
    if (b.graph_pre != nullptr) {
      CUDA_CHECK(cudaGraphLaunch(b.graph_pre, b.stream));
    } else {
      enqueue_pre_eigen(b);
    }

    // Eager: cuSOLVER may do host-side work per call that a captured graph
    // would not replay, so it stays out of the graphs.
    CUSOLVER_CHECK(cusolverDnXsyevBatched(
        b.cusolver_handle, b.cusolver_params, cusolver_jobz, cusolver_uplo,
        T::NR_RECEIVERS, CUDA_C_32F,
        reinterpret_cast<void *>(b.decomp_visibilities.get()), T::NR_RECEIVERS,
        CUDA_R_32F, reinterpret_cast<void *>(b.eigenvalues.get()), CUDA_C_32F,
        b.cusolver_work_device.get(), b.cusolver_work_device_size,
        b.cusolver_work_host, b.cusolver_work_host_size, b.cusolver_info.get(),
        CUSOLVER_BATCH_SIZE));

    // The post-eigen section's first op (accumulate_visibilities) adds into
    // d_visibilities_accumulator. Wait for dump_visibilities' last reset of
    // that accumulator (on buffers[0].stream) to land first -- a GPU-side
    // wait, not a host-blocking sync. No-op until the first dump happens.
    cudaStreamWaitEvent(b.stream, visibilities_reset_done, 0);

    // Post-eigendecomposition section: accumulation, beamforming GEMM and
    // output-layout permutations — again one graph launch when available.
    if (b.graph_post != nullptr) {
      CUDA_CHECK(cudaGraphLaunch(b.graph_post, b.stream));
    } else {
      enqueue_post_eigen(b);
    }

    // Lets dump_visibilities (on buffers[0].stream) wait for this buffer's
    // accumulate_visibilities to land before reading
    // d_visibilities_accumulator, without a cudaDeviceSynchronize().
    cudaEventRecord(b.accumulate_done, b.stream);

    CUFFT_CHECK(cufftXtExec(b.fft_plan, (void *)b.samples_cufft_input.get(),
                            (void *)b.samples_cufft_output.get(),
                            CUFFT_FORWARD));

    detect_and_downsample_fft_launch(
        (float2 *)b.samples_cufft_output.get(),
        (float *)b.cufft_downsampled_output.get(), T::NR_CHANNELS,
        T::NR_POLARIZATIONS,
        T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION,
        T::NR_BEAMS, T::FFT_DOWNSAMPLE_FACTOR, b.stream);
    cudaEventRecord(stop_run[benchmark_runs_done], b.stream);

    // Output handling
    if (output_ != nullptr && !dummy_run) {
      size_t block_num =
          output_->register_beam_data_block(start_seq_num, end_seq_num);
      size_t eigenvalue_block_num =
          output_->register_eigendecomposition_data_block(start_seq_num,
                                                          end_seq_num);
      size_t fft_block_num =
          output_->register_fft_block(start_seq_num, end_seq_num);

      if (block_num != std::numeric_limits<size_t>::max()) {
        void *landing_pointer =
            output_->get_beam_data_landing_pointer(block_num);
        cudaMemcpyAsync(landing_pointer, b.beamformer_data_output_half.get(),
                        sizeof(HalfBeamformerOutput), cudaMemcpyDefault,
                        b.stream);
        auto *output_ctx = new OutputTransferCompleteContext{
            .output = this->output_, .block_index = block_num};
        cudaLaunchHostFunc(b.stream, output_transfer_complete_host_func,
                           output_ctx);
        bool *arrivals_output_pointer =
            (bool *)output_->get_arrivals_data_landing_pointer(block_num);
        std::memcpy(arrivals_output_pointer, packet_data->get_arrivals_ptr(),
                    packet_data->get_arrivals_size());
        output_->register_arrivals_transfer_complete(block_num);
      }

      if (eigenvalue_block_num != std::numeric_limits<size_t>::max()) {

        void *eigenvalues_output_pointer =
            (void *)output_->get_eigenvalues_data_landing_pointer(
                eigenvalue_block_num);

        void *eigenvectors_output_pointer =
            (void *)output_->get_eigenvectors_data_landing_pointer(
                eigenvalue_block_num);

        cudaMemcpyAsync(eigenvalues_output_pointer, b.eigenvalues.get(),
                        sizeof(Eigenvalues), cudaMemcpyDefault, b.stream);

        cudaMemcpyAsync(
            eigenvectors_output_pointer, b.decomp_visibilities.get(),
            sizeof(DecompositionVisibilities), cudaMemcpyDefault, b.stream);

        auto *eig_output_ctx = new OutputTransferCompleteContext{
            .output = this->output_, .block_index = eigenvalue_block_num};
        cudaLaunchHostFunc(b.stream, eigen_output_transfer_complete_host_func,
                           eig_output_ctx);
      }

      if (fft_block_num != std::numeric_limits<size_t>::max()) {
        auto *fft_output_pointer =
            (void *)output_->get_fft_landing_pointer(fft_block_num);
        cudaMemcpyAsync(fft_output_pointer, b.cufft_downsampled_output.get(),
                        sizeof(typename T::FFTOutputType), cudaMemcpyDefault,
                        b.stream);

        auto *fft_output_ctx = new OutputTransferCompleteContext{
            .output = this->output_, .block_index = fft_block_num};
        cudaLaunchHostFunc(b.stream, fft_output_transfer_complete_host_func,
                           fft_output_ctx);
      }
      num_correlation_units_integrated += 1;
      if (num_correlation_units_integrated >=
          NR_CORRELATED_BLOCKS_TO_ACCUMULATE) {
        dump_visibilities();
      }
    }

    // Rotate buffer indices
    if (!dummy_run) {
      current_buffer = (current_buffer + 1) % num_buffers;
      benchmark_runs_done = (benchmark_runs_done + 1) % NR_BENCHMARKING_RUNS;
    }
  }

  // ---- Static mid-pipeline sections -----------------------------------
  // Everything in these two methods operates on device pointers that are
  // fixed for the lifetime of a PipelineResources, which is what allows
  // them to be captured into CUDA graphs (capture_graph) and replayed as a
  // single launch each instead of ~20 individual launches.  Any op added
  // here must use only per-buffer device pointers — no per-run host
  // pointers — to keep the capture valid.

  // From half-converted samples up to the unpacked hermitian matrices that
  // feed the eigendecomposition.
  void enqueue_pre_eigen(PipelineResources &b) {
    tensor_16.runPermutation("packetToPreAlign", alpha,
                             (__half *)b.samples_half.get(),
                             (__half *)b.samples_pre_align.get(), b.stream);

    apply_delays_launch((__half *)b.samples_pre_align.get(),
                        (__half *)b.samples_aligned.get(), d_subpacket_delays,
                        T::NR_RECEIVERS_PER_PACKET, T::NR_FPGA_SOURCES,
                        T::NR_PACKETS_FOR_CORRELATION, T::NR_POLARIZATIONS,
                        T::NR_CHANNELS, T::NR_TIME_STEPS_PER_PACKET, b.stream);

    aligned_to_corr_input<T::NR_CHANNELS, T::NR_POLARIZATIONS, T::NR_RECEIVERS,
                          T::NR_RECEIVERS_PER_PACKET,
                          T::NR_TIME_STEPS_PER_PACKET,
                          T::NR_PACKETS_FOR_CORRELATION,
                          T::NR_PADDED_RECEIVERS, NR_TIMES_PER_BLOCK>(
        (__half *)b.samples_aligned.get(), (__half *)b.correlator_input.get(),
        b.stream);

    correlator.launchAsync((CUstream)b.stream,
                           (CUdeviceptr)b.correlator_output.get(),
                           (CUdeviceptr)b.correlator_input.get());

    // Fuses visCorrToBaseline + D2D trim + visBaselineTrimmedToTrimmed into
    // one kernel pass (saves 3 cuTensor launches + 1 D2D copy).  No atomic
    // accumulation here -- accumulate_visibilities stays in enqueue_post_eigen
    // where cuSOLVER naturally staggers concurrent buffer access to the shared
    // accumulator and avoids contention.
    corr_to_trimmed((float *)b.correlator_output.get(),
                    (float *)b.visibilities_trimmed.get(), T::NR_CHANNELS,
                    NR_BASELINES, NR_UNPADDED_BASELINES,
                    T::NR_POLARIZATIONS * T::NR_POLARIZATIONS * 2, b.stream);

    tensor_32.runPermutation("visCorrToDecomp", alpha_32,
                             (float *)b.visibilities_trimmed.get(),
                             (float *)b.visibilities_permuted.get(), b.stream);
    unpack_triangular_baseline_batch_launch<cuComplex>(
        (cuComplex *)b.visibilities_permuted.get(),
        (cuComplex *)b.decomp_visibilities.get(), T::NR_RECEIVERS,
        T::NR_CHANNELS * T::NR_POLARIZATIONS * T::NR_POLARIZATIONS,
        T::NR_CHANNELS, b.stream);
  }

  // From visibility accumulation through beamforming to the cuFFT input
  // layout.  The cuSOLVER eigendecomposition sits between the two sections
  // and stays eager (it may do host-side work per call that stream capture
  // would not replay); cuFFT + downsample stay eager after this section for
  // the same caution.
  void enqueue_post_eigen(PipelineResources &b) {
    accumulate_visibilities((float *)b.visibilities_trimmed.get(),
                            (float *)d_visibilities_accumulator,
                            2 * NR_UNPADDED_BASELINES * T::NR_POLARIZATIONS *
                                T::NR_POLARIZATIONS * T::NR_CHANNELS,
                            b.stream);

    aligned_to_col_maj_cons<T::NR_CHANNELS, T::NR_POLARIZATIONS,
                            T::NR_RECEIVERS, T::NR_RECEIVERS_PER_PACKET,
                            T::NR_TIME_STEPS_PER_PACKET,
                            T::NR_PACKETS_FOR_CORRELATION>(
        (__half *)b.samples_aligned.get(),
        (__half *)b.samples_consolidated_col_maj.get(), b.stream);

    update_weights((__half *)b.weights.get(), (__half *)b.weights_updated.get(),
                   T::NR_BEAMS, T::NR_RECEIVERS, T::NR_CHANNELS,
                   T::NR_POLARIZATIONS, (float *)b.eigenvalues.get(),
                   (float *)b.visibilities_trimmed.get(), b.stream);

    tensor_16.runPermutation("weightsInputToCCGLIB", alpha,
                             (__half *)b.weights_updated.get(),
                             (__half *)b.weights_permuted.get(), b.stream);

    b.gemm_handle->Run((CUdeviceptr)b.weights_permuted.get(),
                       (CUdeviceptr)b.samples_consolidated_col_maj.get(),
                       (CUdeviceptr)b.beamformer_output.get());

    beam_ccglib_to_half_output<T::NR_CHANNELS, T::NR_POLARIZATIONS,
                               T::NR_BEAMS, NR_TIME_STEPS_FOR_CORRELATION>(
        (float *)b.beamformer_output.get(),
        (__half *)b.beamformer_data_output_half.get(), b.stream);

    tensor_32.runPermutation("beamToCUFFTInput", alpha_32,
                             (float *)b.beamformer_output.get(),
                             (float *)b.samples_cufft_input.get(), b.stream);
  }

  // Capture one section into an instantiated graph.  Raw CUDA calls (not
  // CUDA_CHECK, which exits the process) so a capture-incompatible op
  // degrades to eager execution instead of aborting startup.
  template <typename EnqueueFn>
  bool capture_graph(PipelineResources &b, EnqueueFn &&enqueue,
                     cudaGraphExec_t &exec_out) {
    cudaGraph_t graph = nullptr;
    if (cudaStreamBeginCapture(b.stream, cudaStreamCaptureModeThreadLocal) !=
        cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    bool enqueue_ok = true;
    try {
      enqueue(b);
    } catch (...) {
      // ccglib/cudawrappers throw on capture-incompatible calls.
      enqueue_ok = false;
    }
    const cudaError_t end_err = cudaStreamEndCapture(b.stream, &graph);
    if (!enqueue_ok || end_err != cudaSuccess || graph == nullptr) {
      if (graph != nullptr) {
        cudaGraphDestroy(graph);
      }
      cudaGetLastError();
      return false;
    }
    cudaGraphExec_t exec = nullptr;
    if (cudaGraphInstantiateWithFlags(&exec, graph, 0) != cudaSuccess ||
        exec == nullptr) {
      cudaGraphDestroy(graph);
      cudaGetLastError();
      return false;
    }
    cudaGraphDestroy(graph);
    exec_out = exec;
    return true;
  }

  LambdaGPUPipeline(const int num_buffers, BeamWeightsT<T> *h_weights,
                    BeamSteering<T> beam_steering,
                    int nr_blocks_to_accumulate = T::NR_CORRELATED_BLOCKS_TO_ACCUMULATE)

      : num_buffers(num_buffers),
        NR_CORRELATED_BLOCKS_TO_ACCUMULATE(nr_blocks_to_accumulate),
        h_weights(h_weights),
        beam_steering_(std::move(beam_steering)),
        correlator(cu::Device(0), tcc::Format::fp16, T::NR_PADDED_RECEIVERS,
                   T::NR_CHANNELS,
                   NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK,
                   T::NR_POLARIZATIONS, std::nullopt,
                   T::NR_PADDED_RECEIVERS_PER_BLOCK,
                   TCC_THREAD_BLOCKS_PER_SM),
        tensor_16(extent, CUTENSOR_R_16F, 128),
        tensor_32(extent, CUTENSOR_R_32F, 128),
        cusolver_jobz(CUSOLVER_EIG_MODE_VECTOR),
        cusolver_uplo(CUBLAS_FILL_MODE_UPPER)

  {
    std::cout << "Correlator instantiated with NR_CHANNELS: " << T::NR_CHANNELS
              << ", NR_RECEIVERS: " << T::NR_PADDED_RECEIVERS
              << ", NR_POLARIZATIONS: " << T::NR_POLARIZATIONS
              << ", NR_SAMPLES_PER_CHANNEL: "
              << NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK
              << ", NR_TIMES_PER_BLOCK: " << NR_TIMES_PER_BLOCK
              << ", NR_BLOCKS_FOR_CORRELATION: " << NR_BLOCKS_FOR_CORRELATION
              << ", NR_RECEIVERS_PER_BLOCK: "
              << T::NR_PADDED_RECEIVERS_PER_BLOCK << std::endl;

    const long long CUFFT_FFT_SIZE = NR_TIME_STEPS_FOR_CORRELATION;
    long long N[] = {CUFFT_FFT_SIZE};
    const size_t NUM_TOTAL_BATCHES =
        T::NR_BEAMS * T::NR_CHANNELS * T::NR_POLARIZATIONS;

    CUDA_CHECK(cudaMalloc((void **)&d_visibilities_accumulator,
                          sizeof(TrimmedVisibilities)));
    CUDA_CHECK(cudaEventCreateWithFlags(&visibilities_reset_done,
                                        cudaEventDisableTiming));

    CUDA_CHECK(cudaMalloc((void **)&d_subpacket_delays,
                          sizeof(int) * T::NR_FPGA_SOURCES));
    CUDA_CHECK(
        cudaMemset(d_subpacket_delays, 0, sizeof(int) * T::NR_FPGA_SOURCES));

    CUDA_CHECK(cudaMalloc((void **)&d_gains, sizeof(typename T::AntennaGains)));
    auto default_gains = get_default_gains<T::NR_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));

    last_frame_processed = 0;
    num_integrated_units_processed = 0;
    num_correlation_units_integrated = 0;
    current_buffer = 0;
    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);

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
    tensor_16.addTensor(modePacketAligned, "aligned");
    tensor_16.addTensor(modePacketAlignedFlat, "alignedFlat");
    tensor_16.addTensor(modePacketPadding, "packet_padding");
    tensor_16.addTensor(modePacketPadded, "packet_padded");
    tensor_16.addTensor(modeCorrelatorInput, "corr_input");
    tensor_16.addTensor(modePlanar, "planar");
    tensor_16.addTensor(modePlanarCons, "planarCons");
    tensor_16.addTensor(modePlanarColMajCons, "planarColMajCons");
    tensor_32.addTensor(modeCUFFTInput, "cufftInput");

    tensor_16.addTensor(modeWeightsInput, "weightsInput");
    tensor_16.addTensor(modeWeightsCCGLIB, "weightsCCGLIB");
    tensor_32.addTensor(modeVisCorr, "visCorr");
    tensor_32.addTensor(modeVisDecomp, "visDecomp");
    tensor_32.addTensor(modeVisCorrBaseline, "visBaseline");
    tensor_32.addTensor(modeVisCorrBaselineTrimmed, "visBaselineTrimmed");
    tensor_32.addTensor(modeVisCorrTrimmed, "visCorrTrimmed");

    tensor_32.addTensor(modeBeamCCGLIB, "beamCCGLIB");
    tensor_32.addTensor(modeBeamOutput, "beamOutput");

    tensor_16.addPermutation("aligned", "packet_padding",
                             CUTENSOR_COMPUTE_DESC_16F, "alignedToPadding");
    tensor_16.addPermutation("packet", "prealign", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToPreAlign");
    tensor_16.addPermutation("packet_padded", "corr_input",
                             CUTENSOR_COMPUTE_DESC_16F, "paddedToCorrInput");
    tensor_16.addPermutation("aligned", "planar", CUTENSOR_COMPUTE_DESC_16F,
                             "alignedToPlanar");
    tensor_16.addPermutation("planarCons", "planarColMajCons",
                             CUTENSOR_COMPUTE_DESC_16F, "consToColMajCons");
    tensor_16.addPermutation("alignedFlat", "planarColMajCons",
                             CUTENSOR_COMPUTE_DESC_16F, "alignedToColMajCons");
    tensor_16.addPermutation("weightsInput", "weightsCCGLIB",
                             CUTENSOR_COMPUTE_DESC_16F, "weightsInputToCCGLIB");
    tensor_32.addPermutation("visCorrTrimmed", "visDecomp",
                             CUTENSOR_COMPUTE_DESC_32F, "visCorrToDecomp");
    tensor_32.addPermutation("beamCCGLIB", "beamOutput",
                             CUTENSOR_COMPUTE_DESC_32F, "beamCCGLIBToOutput");
    tensor_32.addPermutation("visCorr", "visBaseline",
                             CUTENSOR_COMPUTE_DESC_32F, "visCorrToBaseline");
    tensor_32.addPermutation("visBaselineTrimmed", "visCorrTrimmed",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "visBaselineTrimmedToTrimmed");
    tensor_32.addPermutation("beamCCGLIB", "cufftInput",
                             CUTENSOR_COMPUTE_DESC_32F, "beamToCUFFTInput");

    buffers.reserve(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(cu_device, work_size);

      // Finalize cuFFT plan for this buffer
      auto &b = buffers.back();
      CUDA_CHECK(
          cudaEventCreateWithFlags(&b.accumulate_done, cudaEventDisableTiming));
      CUDA_CHECK(cudaMemsetAsync(b.correlator_input.get(), 0,
                                 sizeof(CorrelatorInput), b.stream));
      CUFFT_CHECK(cufftXtMakePlanMany(b.fft_plan, 1, N, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NULL, 1, CUFFT_FFT_SIZE,
                                      CUDA_C_32F, NUM_TOTAL_BATCHES, &work_size,
                                      CUDA_C_32F));
      CUFFT_CHECK(cufftSetStream(b.fft_plan, b.stream));
      CUFFT_CHECK(cufftSetWorkArea(b.fft_plan, b.cufft_work_area.get()));

      // Copy initial weights
      cudaMemcpyAsync(b.weights.get(), h_weights, sizeof(BeamWeights),
                      cudaMemcpyDefault, b.stream);
      tensor_16.runPermutation("weightsInputToCCGLIB", alpha,
                               (__half *)b.weights.get(),
                               (__half *)b.weights_permuted.get(), b.stream);
      // Registered before the warmup run below, so the first (always-overdue)
      // maybe_refresh() steers every buffer in one shot.
      beam_steering_.register_buffer(b.weights.get(), b.stream);
    }
    for (auto i = 0; i < NR_BENCHMARKING_RUNS; ++i) {
      cudaEventCreate(&start_run[i]);
      cudaEventCreate(&stop_run[i]);
    }

    cudaDeviceSynchronize();

    // Warm up the pipeline *before* attempting graph capture. This is the
    // first-ever call to most cuTENSOR permutations, the TCC correlator, and
    // the ccglib GEMM (only "weightsInputToCCGLIB" has run so far, above).
    // Each of these libraries may do one-time lazy initialization on first
    // use -- NVRTC JIT compilation, module loading, on-disk JIT-cache
    // writes -- which is not safe to perform while cudaStreamBeginCapture is
    // active (can hang or segfault, especially with a cold cache). Running
    // this eagerly first ensures capture below only ever records
    // already-loaded kernels. Because everything is zeroed it should have
    // negligible effect on output.
    typename T::PacketFinalDataType warmup_packet;
    std::memset(warmup_packet.samples, 0,
                warmup_packet.get_samples_elements_size());
    std::memset(warmup_packet.scales, 0,
                warmup_packet.get_scales_element_size());
    std::memset(warmup_packet.arrivals, 0, warmup_packet.get_arrivals_size());
    execute_pipeline(&warmup_packet, true);
    cudaDeviceSynchronize();

    // Capture the two static mid-pipeline sections of each buffer into CUDA
    // graphs (~21 launches collapse into 2 per run).  Any failure falls back
    // to eager execution for all buffers — functionally identical, just more
    // launch overhead.  SPATIAL_DISABLE_CUDA_GRAPH=1 forces the eager path.
    if (std::getenv("SPATIAL_DISABLE_CUDA_GRAPH") == nullptr) {
      bool all_ok = true;
      for (auto &b : buffers) {
        if (!capture_graph(
                b, [this](PipelineResources &r) { enqueue_pre_eigen(r); },
                b.graph_pre) ||
            !capture_graph(
                b, [this](PipelineResources &r) { enqueue_post_eigen(r); },
                b.graph_post)) {
          all_ok = false;
          break;
        }
      }
      if (!all_ok) {
        for (auto &b : buffers) {
          if (b.graph_pre) {
            cudaGraphExecDestroy(b.graph_pre);
            b.graph_pre = nullptr;
          }
          if (b.graph_post) {
            cudaGraphExecDestroy(b.graph_post);
            b.graph_post = nullptr;
          }
        }
        WARN_LOG("CUDA graph capture failed — pipeline will run eagerly");
      } else {
        INFO_LOG("CUDA graphs captured for {} pipeline buffers",
                 buffers.size());
        // Exercise the captured graphs (cudaGraphLaunch replay) once before
        // real traffic arrives, so a replay problem surfaces here rather
        // than on the first live buffer.
        execute_pipeline(&warmup_packet, true);
      }
      cudaDeviceSynchronize();
    }

    // these need to be set after the dummy run(s).
    visibilities_start_seq_num = -1;
    visibilities_end_seq_num = -1;
    visibilities_missing_packets = 0;
  };
  ~LambdaGPUPipeline() {
    // If there are visibilities in the accumulator on the GPU - dump them
    // out to disk. These will get tagged with a -1 end_seq_id currently
    // which is not fully ideal.

    if (visibilities_reset_done)
      cudaEventDestroy(visibilities_reset_done);
  };
  virtual void set_subpacket_delays(int *delays_subpacket) override {
    subpacket_delays_ = delays_subpacket;
    CUDA_CHECK(cudaMemcpy(d_subpacket_delays, subpacket_delays_,
                          sizeof(int) * T::NR_FPGA_SOURCES, cudaMemcpyDefault));
  }

  virtual void set_antenna_gains(std::complex<float> *gains) override {
    std::cout << "setting antenna gains on LambdaGPUPipeline...\n";
    gains_ = gains;
    CUDA_CHECK(cudaMemcpy(d_gains, gains, sizeof(typename T::AntennaGains),
                          cudaMemcpyDefault));

    std::cout << "Loaded gains are:\n";
    for (auto i = 0; i < T::NR_CHANNELS; ++i) {
      for (auto j = 0; j < T::NR_POLARIZATIONS; ++j) {
        for (auto k = 0; k < T::NR_RECEIVERS; ++k) {
          std::cout << "channel " << i << " pol " << j << " receiver " << k
                    << " val "
                    << gains[i * T::NR_POLARIZATIONS * T::NR_RECEIVERS +
                             j * T::NR_RECEIVERS + k]
                           .real()
                    << " + "
                    << gains[i * T::NR_POLARIZATIONS * T::NR_RECEIVERS +
                             j * T::NR_RECEIVERS + k]
                           .imag()
                    << "j.\n";
        }
      }
    }
    cudaDeviceSynchronize();
    std::cout << "gains uploaded successfully...\n";
  }

  void dump_visibilities(const uint64_t end_seq_num = 0) override {

    INFO_LOG("Dumping correlations to host...");
    int current_num_integrated_units_processed =
        num_correlation_units_integrated;
    INFO_LOG("Current num integrated units processed is {}",
             current_num_integrated_units_processed);
    // GPU-side wait (not a host-blocking sync): make buffers[0].stream's
    // upcoming memcpy of d_visibilities_accumulator wait for every buffer's
    // most recent accumulate_visibilities to land first. accumulate_done is a
    // no-op wait until it has been recorded at least once.
    for (auto &buf : buffers) {
      cudaStreamWaitEvent(buffers[0].stream, buf.accumulate_done, 0);
    }
    const int visibilities_total_packets =
        current_num_integrated_units_processed *
        visibilities_total_packets_per_block;
    size_t block_num = output_->register_visibilities_block(
        visibilities_start_seq_num, end_seq_num, visibilities_missing_packets,
        visibilities_total_packets);
    visibilities_start_seq_num = -1;
    visibilities_missing_packets = 0;
    if (block_num != std::numeric_limits<size_t>::max()) {
      void *landing_pointer =
          output_->get_visibilities_landing_pointer(block_num);
      cudaMemcpyAsync(landing_pointer, d_visibilities_accumulator,
                      sizeof(TrimmedVisibilities), cudaMemcpyDefault,
                      buffers[0].stream);
      auto *output_ctx = new OutputTransferCompleteContext{
          .output = this->output_, .block_index = block_num};

      cudaLaunchHostFunc(buffers[0].stream,
                         output_visibilities_transfer_complete_host_func,
                         output_ctx);
    }
    cudaMemsetAsync(d_visibilities_accumulator, 0, sizeof(TrimmedVisibilities),
                    buffers[0].stream);
    num_correlation_units_integrated.store(0);
    // Record the reset so every buffer's next accumulate_visibilities (via
    // the wait above, in execute_pipeline) is ordered after it.
    cudaEventRecord(visibilities_reset_done, buffers[0].stream);
  };
};

