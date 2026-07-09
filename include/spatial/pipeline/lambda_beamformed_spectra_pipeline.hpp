#pragma once

template <typename T>
class LambdaBeamformedSpectraPipeline : public GPUPipeline {
private:
  static constexpr int NR_TIMES_PER_BLOCK = 128 / 16; // NR_BITS;

  static constexpr int NR_BLOCKS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET /
      NR_TIMES_PER_BLOCK;
  static constexpr int NR_TIME_STEPS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET;
  static constexpr int COMPLEX = 2;

  using FFTCUFFTInputType =
      float2[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
            [T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION];
  using FFTCUFFTOutputType =
      float2[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
            [T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION];
  using FFTOutputType =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
           [T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION /
            T::FFT_DOWNSAMPLE_FACTOR];
  using BeamformerOutput =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
           [NR_TIME_STEPS_FOR_CORRELATION][COMPLEX];
  using BeamWeights = BeamWeightsT<T>;
  struct PipelineResources {
    cudaStream_t stream;
    cudaStream_t host_stream;
    ManagedCufftPlan fft_plan;

    DevicePtr<typename T::InputPacketSamplesType> samples_entry;
    DevicePtr<typename T::PacketScalesType> scales;
    DevicePtr<typename T::HalfPacketSamplesType> samples_half,
        samples_consolidated, samples_consolidated_col_maj;
    DevicePtr<FFTCUFFTInputType> samples_cufft_input;
    DevicePtr<FFTCUFFTOutputType> samples_cufft_output;
    DevicePtr<FFTOutputType> cufft_downsampled_output;
    DevicePtr<BeamWeights> weights;
    DevicePtr<BeamWeights> weights_permuted;
    DevicePtr<BeamformerOutput> beamformer_output;
    DevicePtr<void> cufft_work_area;

    std::unique_ptr<ccglib::mma::GEMM> gemm_handle;

    PipelineResources(CUdevice cu_device, size_t work_size)
        : samples_entry(make_device_ptr<typename T::InputPacketSamplesType>()),
          scales(make_device_ptr<typename T::PacketScalesType>()),
          samples_half(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_consolidated(
              make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_consolidated_col_maj(
              make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_cufft_input(make_device_ptr<FFTCUFFTInputType>()),
          samples_cufft_output(make_device_ptr<FFTCUFFTOutputType>()),
          cufft_downsampled_output(make_device_ptr<FFTOutputType>()),
          weights(make_device_ptr<BeamWeights>()),
          weights_permuted(make_device_ptr<BeamWeights>()),
          beamformer_output(make_device_ptr<BeamformerOutput>()),
          cufft_work_area(make_device_ptr<void>(work_size)) {
      // Stream Creation
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUDA_CHECK(
          cudaStreamCreateWithFlags(&host_stream, cudaStreamNonBlocking));

      // GEMM Initialization
      gemm_handle = std::make_unique<ccglib::mma::GEMM>(
          T::NR_CHANNELS * T::NR_POLARIZATIONS, T::NR_BEAMS,
          NR_TIMES_PER_BLOCK * NR_BLOCKS_FOR_CORRELATION, T::NR_RECEIVERS,
          cu_device, stream, ccglib::ValueType::float16, ccglib::mma::basic);
    }

    ~PipelineResources() {
      cudaStreamDestroy(stream);
      cudaStreamDestroy(host_stream);
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
          gemm_handle(std::move(other.gemm_handle)) {
      other.stream = nullptr;
      other.host_stream = nullptr;
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

  int num_buffers;
  std::vector<PipelineResources> buffers;

  typename T::AntennaGains *d_gains;
  // We are converting it to fp16 so this should not be changable anymore.

  inline static const __half alpha = __float2half(1.0f);

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

  inline static const std::vector<int> modePacket{'c', 'o', 'f', 'u',
                                                  'n', 'p', 'z'};
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

  inline static const std::vector<int> modeBeamCCGLIB{'c', 'p', 'z', 'm', 's'};
  inline static const std::vector<int> modeWeightsInput{'c', 'p', 'm', 'r',
                                                        'z'};
  inline static const std::vector<int> modeWeightsCCGLIB{'c', 'p', 'z', 'm',
                                                         'r'};

  inline static const std::unordered_map<int, int64_t> extent = {

      {'b', NR_BLOCKS_FOR_CORRELATION},
      {'c', T::NR_CHANNELS},
      {'f', T::NR_FPGA_SOURCES},
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

  BeamWeights *h_weights;
  // See the comment on this member in LambdaGPUPipeline (pipeline.hpp above)
  // -- same helper, same inert-when-unsteered contract.
  BeamSteering<T> beam_steering_;

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

    tensor_16.runPermutation("packetToPlanar", alpha,
                             (__half *)b.samples_half.get(),
                             (__half *)b.samples_consolidated.get(), b.stream);

    tensor_16.runPermutation(
        "consToColMajCons", alpha, (__half *)b.samples_consolidated.get(),
        (__half *)b.samples_consolidated_col_maj.get(), b.stream);

    // this only needs to be run once.
    tensor_16.runPermutation("weightsInputToCCGLIB", alpha,
                             (__half *)b.weights.get(),
                             (__half *)b.weights_permuted.get(), b.stream);

    b.gemm_handle->Run((CUdeviceptr)b.weights_permuted.get(),
                       (CUdeviceptr)b.samples_consolidated_col_maj.get(),
                       (CUdeviceptr)b.beamformer_output.get());

    tensor_32.runPermutation("beamToCUFFTInput", alpha_32,
                             (float *)b.beamformer_output.get(),
                             (float *)b.samples_cufft_input.get(), b.stream);

    CUFFT_CHECK(cufftXtExec(b.fft_plan, (void *)b.samples_cufft_input.get(),
                            (void *)b.samples_cufft_output.get(),
                            CUFFT_FORWARD));

    detect_and_downsample_fft_launch(
        (float2 *)b.samples_cufft_output.get(),
        (float *)b.cufft_downsampled_output.get(), T::NR_CHANNELS,
        T::NR_POLARIZATIONS,
        T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION,
        T::NR_BEAMS, T::FFT_DOWNSAMPLE_FACTOR, b.stream);
    if (output_ != nullptr && !dummy_run) {
      // -1, -1 is required but not used. Interface allows for single channel /
      // pol to be passed but this implementation does not use it.
      size_t fft_block_num =
          output_->register_fft_block(start_seq_num, end_seq_num);
      // size_t::max means no FFT writer attached -- the landing pointer
      // would be nullptr.
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
    }

    // Rotate buffer indices
    if (!dummy_run) {
      current_buffer = (current_buffer + 1) % num_buffers;
    }
  }
  LambdaBeamformedSpectraPipeline(const int num_buffers,
                                  BeamWeightsT<T> *h_weights,
                                  BeamSteering<T> beam_steering)

      : num_buffers(num_buffers), h_weights(h_weights),
        beam_steering_(std::move(beam_steering)),
        tensor_16(extent, CUTENSOR_R_16F, 128),
        tensor_32(extent, CUTENSOR_R_32F, 128)

  {
    std::cout << "Beamformed Spectra instantiated with NR_CHANNELS: "
              << T::NR_CHANNELS << ", NR_RECEIVERS: " << T::NR_RECEIVERS
              << ", NR_POLARIZATIONS: " << T::NR_POLARIZATIONS
              << ", NR_SAMPLES_PER_CHANNEL: "
              << NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK
              << ", NR_TIMES_PER_BLOCK: " << NR_TIMES_PER_BLOCK
              << ", NR_BLOCKS_FOR_FFT: " << NR_BLOCKS_FOR_CORRELATION
              << ", NR_BEAMS: " << T::NR_BEAMS << std::endl;

    const long long CUFFT_FFT_SIZE = NR_TIME_STEPS_FOR_CORRELATION;
    long long N[] = {CUFFT_FFT_SIZE};
    const size_t NUM_TOTAL_BATCHES =
        T::NR_BEAMS * T::NR_CHANNELS * T::NR_POLARIZATIONS;

    CUDA_CHECK(cudaMalloc((void **)&d_gains, sizeof(typename T::AntennaGains)));
    auto default_gains = get_default_gains<T::NR_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));

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

    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);
    buffers.reserve(num_buffers);
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
      cudaMemcpy(b.weights.get(), h_weights, sizeof(BeamWeights),
                 cudaMemcpyDefault);
      // Registered before the warmup run below, so the first (always-overdue)
      // maybe_refresh() steers every buffer in one shot.
      beam_steering_.register_buffer(b.weights.get(), b.stream);
    }
    last_frame_processed = 0;
    current_buffer = 0;
    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modePlanar, "planar");
    tensor_16.addTensor(modePlanarCons, "planarCons");
    tensor_16.addTensor(modePlanarColMajCons, "planarColMajCons");

    tensor_16.addTensor(modeWeightsInput, "weightsInput");
    tensor_16.addTensor(modeWeightsCCGLIB, "weightsCCGLIB");

    tensor_32.addTensor(modeCUFFTInput, "cufftInput");
    tensor_32.addTensor(modeBeamCCGLIB, "beamCCGLIB");
    tensor_16.addPermutation("packet", "planar", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToPlanar");
    tensor_16.addPermutation("planarCons", "planarColMajCons",
                             CUTENSOR_COMPUTE_DESC_16F, "consToColMajCons");
    tensor_16.addPermutation("weightsInput", "weightsCCGLIB",
                             CUTENSOR_COMPUTE_DESC_16F, "weightsInputToCCGLIB");
    tensor_32.addPermutation("beamCCGLIB", "cufftInput",
                             CUTENSOR_COMPUTE_DESC_32F, "beamToCUFFTInput");
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
};
