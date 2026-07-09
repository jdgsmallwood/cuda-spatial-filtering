#pragma once

template <typename T> class LambdaAntennaSpectraPipeline : public GPUPipeline {

private:
  int num_buffers;
  std::vector<cudaStream_t> streams;

  typename T::AntennaGains *d_gains;
  // We are converting it to fp16 so this should not be changable anymore.
  static constexpr int NR_TIMES_PER_BLOCK = 128 / 16; // NR_BITS;

  static constexpr int NR_BLOCKS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET /
      NR_TIMES_PER_BLOCK;
  static constexpr int NR_TIME_STEPS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET;
  static constexpr int COMPLEX = 2;

  inline static const __half alpha = __float2half(1.0f);

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
  // o and u need to end up together and will be interpreted as b x t in the
  // next transformation. Similarly f x n = r in next transformation.
  inline static const std::vector<int> modeCUFFTInput{'c', 'p', 'f', 'n',
                                                      'o', 'u', 'z'};
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

  int current_buffer;
  std::atomic<int> last_frame_processed;

  std::vector<typename T::InputPacketSamplesType *> d_samples_entry;
  std::vector<typename T::HalfPacketSamplesType *> d_samples_half,
      d_samples_padding;
  std::vector<typename T::FFTCUFFTPreprocessingType *>
      d_samples_cufft_preprocessing;
  std::vector<typename T::MultiChannelFFTCUFFTInputType *>
      d_samples_cufft_input;
  std::vector<typename T::MultiChannelFFTCUFFTOutputType *>
      d_samples_cufft_output;
  std::vector<typename T::MultiChannelAntennaFFTOutputType *>
      d_cufft_downsampled_output;
  std::vector<typename T::PacketScalesType *> d_scales;

  static constexpr int fft_total_packets_per_block =
      T::NR_CHANNELS * T::NR_PACKETS_FOR_CORRELATION * T::NR_FPGA_SOURCES;
  int fft_missing_packets;
  std::vector<cufftHandle> fft_plan;
  std::vector<void *> d_cufft_work_area;

public:
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {

    const uint64_t start_seq_num = packet_data->start_seq_id;
    const uint64_t end_seq_num = packet_data->end_seq_id;
    INFO_LOG("Pipeline run started with start_seq {} and end seq {}",
             start_seq_num, end_seq_num);

    // dummy_run must be forwarded so the warmup run's release_buffer host
    // func skips the release (state_ is unset during construction).
    LambdaPipelineIngest<T>::ingest_and_scale(
        this->state_, packet_data, streams[current_buffer],
        streams[current_buffer], d_samples_entry[current_buffer],
        d_scales[current_buffer], d_gains, d_samples_half[current_buffer],
        dummy_run);

    tensor_16.runPermutation(
        "packetToCUFFTInput", alpha, (__half *)d_samples_half[current_buffer],
        (__half *)d_samples_cufft_preprocessing[current_buffer],
        streams[current_buffer]);

    // convert to float
    get_data_for_multi_channel_fft_launch<
        typename T::FFTCUFFTPreprocessingType,
        typename T::MultiChannelFFTCUFFTInputType>(
        (typename T::FFTCUFFTPreprocessingType *)
            d_samples_cufft_preprocessing[current_buffer],
        d_samples_cufft_input[current_buffer], T::NR_CHANNELS,
        T::NR_POLARIZATIONS, NR_TIME_STEPS_FOR_CORRELATION, T::NR_RECEIVERS,
        streams[current_buffer]);

    CUFFT_CHECK(cufftXtExec(
        fft_plan[current_buffer], (void *)d_samples_cufft_input[current_buffer],
        (void *)d_samples_cufft_output[current_buffer], CUFFT_FORWARD));

    detect_and_downsample_multi_channel_fft_launch<
        typename T::MultiChannelFFTCUFFTOutputType,
        typename T::MultiChannelAntennaFFTOutputType>(
        d_samples_cufft_output[current_buffer],
        d_cufft_downsampled_output[current_buffer], T::NR_CHANNELS,
        T::NR_POLARIZATIONS,
        T::NR_TIME_STEPS_PER_PACKET * T::NR_PACKETS_FOR_CORRELATION,
        T::NR_RECEIVERS, T::FFT_DOWNSAMPLE_FACTOR, streams[current_buffer]);
    // Output handling
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
        cudaMemcpyAsync(fft_output_pointer,
                        d_cufft_downsampled_output[current_buffer],
                        sizeof(typename T::MultiChannelAntennaFFTOutputType),
                        cudaMemcpyDefault, streams[current_buffer]);

        auto *fft_output_ctx = new OutputTransferCompleteContext{
            .output = this->output_, .block_index = fft_block_num};
        cudaLaunchHostFunc(streams[current_buffer],
                           fft_output_transfer_complete_host_func,
                           fft_output_ctx);
      }
    }

    // Rotate buffer indices
    if (!dummy_run) {
      current_buffer = (current_buffer + 1) % num_buffers;
    }
  }
  LambdaAntennaSpectraPipeline(const int num_buffers)

      : num_buffers(num_buffers), tensor_16(extent, CUTENSOR_R_16F, 128)

  {
    std::cout << "Spectra Analyzer instantiated with NR_CHANNELS: "
              << T::NR_CHANNELS << ", NR_RECEIVERS: " << T::NR_RECEIVERS
              << ", NR_POLARIZATIONS: " << T::NR_POLARIZATIONS
              << ", NR_SAMPLES_PER_CHANNEL: "
              << NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK
              << ", NR_TIMES_PER_BLOCK: " << NR_TIMES_PER_BLOCK
              << ", NR_BLOCKS_FOR_FFT: " << NR_BLOCKS_FOR_CORRELATION
              << std::endl;

    streams.resize(2 * num_buffers);
    d_samples_entry.resize(num_buffers);
    d_scales.resize(num_buffers);
    d_samples_half.resize(num_buffers);
    d_samples_cufft_input.resize(num_buffers);
    d_samples_cufft_preprocessing.resize(num_buffers);
    d_cufft_downsampled_output.resize(num_buffers);
    d_samples_cufft_output.resize(num_buffers);

    fft_plan.resize(num_buffers);
    d_cufft_work_area.resize(num_buffers);

    for (auto i = 0; i < num_buffers; ++i) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[num_buffers + i],
                                           cudaStreamNonBlocking));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_entry[i],
                            sizeof(typename T::InputPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_half[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_cufft_preprocessing[i],
                            sizeof(typename T::FFTCUFFTPreprocessingType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_cufft_input[i],
                            sizeof(typename T::MultiChannelFFTCUFFTInputType)));
      CUDA_CHECK(
          cudaMalloc((void **)&d_samples_cufft_output[i],
                     sizeof(typename T::MultiChannelFFTCUFFTOutputType)));
      CUDA_CHECK(
          cudaMalloc((void **)&d_cufft_downsampled_output[i],
                     sizeof(typename T::MultiChannelAntennaFFTOutputType)));
      CUDA_CHECK(cudaMalloc((void **)&d_scales[i],
                            sizeof(typename T::PacketScalesType)));
    }

    last_frame_processed = 0;
    current_buffer = 0;
    cudaDeviceSynchronize();
    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modeCUFFTInput, "cufftInput");

    tensor_16.addPermutation("packet", "cufftInput", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToCUFFTInput");

    // set up CUFFT plan for fine-channelization
    const int CUFFT_RANK = 1;
    const long long CUFFT_FFT_SIZE = NR_TIME_STEPS_FOR_CORRELATION;
    long long N[] = {CUFFT_FFT_SIZE};
    const long long CUFFT_ISTRIDE = 1;
    const long long CUFFT_OSTRIDE = 1;
    const long long CUFFT_IDIST = CUFFT_FFT_SIZE;
    const long long CUFFT_ODIST = CUFFT_FFT_SIZE;
    const size_t NUM_TOTAL_BATCHES =
        T::NR_RECEIVERS * T::NR_CHANNELS * T::NR_POLARIZATIONS;
    INFO_LOG("FFT initialized with {} total batches with a {} FFT each run "
             "(RECEIVERS x CHANNELS x POL)",
             NUM_TOTAL_BATCHES, CUFFT_FFT_SIZE);
    size_t work_size = 0;
    cudaDataType input_type = CUDA_C_32F;
    cudaDataType output_type = CUDA_C_32F;
    cudaDataType compute_type = CUDA_C_32F;

    for (int i = 0; i < num_buffers; ++i) {
      CUFFT_CHECK(cufftCreate(&fft_plan[i]));
      CUFFT_CHECK(cufftXtMakePlanMany(
          fft_plan[i], CUFFT_RANK, N, NULL, CUFFT_ISTRIDE, CUFFT_IDIST,
          input_type, NULL, CUFFT_OSTRIDE, CUFFT_ODIST, output_type,
          NUM_TOTAL_BATCHES, &work_size, compute_type));

      CUFFT_CHECK(cufftSetStream(fft_plan[i], streams[i]));
      CUDA_CHECK(cudaMalloc(&d_cufft_work_area[i], work_size));
      CUFFT_CHECK(cufftSetWorkArea(fft_plan[i], d_cufft_work_area[i]));
    }
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
  ~LambdaAntennaSpectraPipeline() {
    // If there are visibilities in the accumulator on the GPU - dump them
    // out to disk. These will get tagged with a -1 end_seq_id currently
    // which is not fully ideal.

    for (auto stream : streams) {
      cudaStreamDestroy(stream);
    }

    for (auto sample : d_samples_entry) {
      cudaFree(sample);
    }

    for (auto scale : d_scales) {
      cudaFree(scale);
    }

    for (auto samples_half : d_samples_half) {
      cudaFree(samples_half);
    }

    for (auto samples_cufft : d_samples_cufft_input) {
      cudaFree(samples_cufft);
    }

    for (auto samples_cufft : d_samples_cufft_preprocessing) {
      cudaFree(samples_cufft);
    }
    for (auto samples_cufft : d_samples_cufft_output) {
      cudaFree(samples_cufft);
    }
    for (auto cufft : d_cufft_downsampled_output) {
      cudaFree(cufft);
    }
  };

  void dump_visibilities(const uint64_t end_seq_num = 0) override {
    // nothing to do.
  }
};

