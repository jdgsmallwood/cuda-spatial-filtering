#pragma once

template <typename T> class LambdaAntennaSpectraPipeline : public GPUPipeline {

private:
  int num_buffers;
  std::vector<cudaStream_t> streams;

  typename T::AntennaGains *d_gains = nullptr;
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
  // c = channel (always NR_FPGA_CHANNELS here -- this pipeline has no post-channelization
  //     cuTensor-permuted tensor, so unlike LambdaGPUPipeline there's no need for a separate
  //     widened-'c'/coarse-'C' split)
  // d = padded receivers
  // f = fpga
  // g = packets for correlation + 2 (pre-alignment padding, dropped by apply_delays_launch)
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
  // Only actually invoked when T::NR_FINE_CHANNELS > 1 -- bridges d_samples_half into the
  // [f,g,u,c,n,p,z] layout apply_delays_launch/reorder_streams_launch/reorder_to_filter_input
  // expect (same "packetToPreAlign" pattern LambdaGPUPipeline uses ahead of channelization; see
  // that file's modePacket/modePacketPreAlign comment for why this reordering step is required
  // rather than feeding raw packet layout straight into reorder_to_filter_input).
  inline static const std::vector<int> modePacketPreAlign{'f', 'g', 'u', 'c',
                                                          'n', 'p', 'z'};
  // o and u need to end up together and will be interpreted as b x t in the
  // next transformation. Similarly f x n = r in next transformation.
  inline static const std::vector<int> modeCUFFTInput{'c', 'p', 'f', 'n',
                                                      'o', 'u', 'z'};
  inline static const std::unordered_map<int, int64_t> extent = {

      {'b', NR_BLOCKS_FOR_CORRELATION},
      {'c', T::NR_FPGA_CHANNELS},
      {'f', T::NR_FPGA_SOURCES},
      {'g', T::NR_PACKETS_FOR_CORRELATION + 2},
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

  // Fine-channelization path (only populated/used when T::NR_FINE_CHANNELS > 1).
  std::vector<typename T::HalfPacketSamplesType *> d_samples_pre_align;
  std::vector<typename T::HalfPacketAlignedSamplesType *> d_samples_aligned,
      d_samples_reordered;
  std::vector<typename FineChannelizer<T>::FilterInputType *> d_channelizer_input;
  std::vector<typename FineChannelizer<T>::FilterOutputType *> d_channelizer_output;
  std::unique_ptr<FineChannelizer<T>> channelizer_;
  // [NR_FPGA_SOURCES], zeroed -- this pipeline never supported delay correction; these tables
  // exist only so apply_delays_launch/reorder_streams_launch can bridge samples_pre_align into
  // reorder_to_filter_input's expected samples_reordered layout as a true no-op reshape.
  int *d_subpacket_delays = nullptr;
  int *d_stream_perm_recv = nullptr, *d_stream_perm_pol = nullptr; // identity by default

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

    if constexpr (T::NR_FINE_CHANNELS > 1) {
      tensor_16.runPermutation(
          "packetToPreAlign", alpha, (__half *)d_samples_half[current_buffer],
          (__half *)d_samples_pre_align[current_buffer],
          streams[current_buffer]);

      apply_delays_launch(
          (__half *)d_samples_pre_align[current_buffer],
          (__half *)d_samples_aligned[current_buffer], d_subpacket_delays,
          T::NR_RECEIVERS_PER_PACKET, T::NR_FPGA_SOURCES,
          T::NR_PACKETS_FOR_CORRELATION, T::NR_POLARIZATIONS,
          T::NR_FPGA_CHANNELS, T::NR_TIME_STEPS_PER_PACKET,
          streams[current_buffer]);

      reorder_streams_launch<T::NR_FPGA_SOURCES, T::NR_PACKETS_FOR_CORRELATION,
                             T::NR_TIME_STEPS_PER_PACKET, T::NR_FPGA_CHANNELS,
                             T::NR_RECEIVERS_PER_PACKET, T::NR_POLARIZATIONS>(
          (__half *)d_samples_aligned[current_buffer],
          (__half *)d_samples_reordered[current_buffer], d_stream_perm_recv,
          d_stream_perm_pol, streams[current_buffer]);

      reorder_to_filter_input<
          T::NR_FPGA_CHANNELS, T::NR_POLARIZATIONS, T::NR_RECEIVERS,
          T::NR_RECEIVERS_PER_PACKET, T::NR_TIME_STEPS_PER_PACKET,
          T::NR_PACKETS_FOR_CORRELATION, FineChannelizer<T>::NR_TAPS,
          T::NR_FINE_CHANNELS>((__half *)d_samples_reordered[current_buffer],
                               (float2 *)d_channelizer_input[current_buffer],
                               streams[current_buffer]);

      channelizer_->launchAsync(streams[current_buffer],
                                d_channelizer_input[current_buffer],
                                d_channelizer_output[current_buffer]);

      channelizer_output_to_antenna_power<
          T::NR_FPGA_CHANNELS, T::NR_FINE_CHANNELS, T::NR_FINE_CHANNEL_EDGE_TRIM,
          T::NR_POLARIZATIONS, T::NR_RECEIVERS,
          FineChannelizer<T>::NR_SAMPLES_PER_FINE_CHANNEL,
          FineChannelizer<T>::NR_TIMES_PER_OUTPUT_BLOCK>(
          (const __half2 *)d_channelizer_output[current_buffer],
          (float *)d_cufft_downsampled_output[current_buffer],
          T::FFT_DOWNSAMPLE_FACTOR, streams[current_buffer]);
    } else {
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
    }
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
              << T::NR_CHANNELS << ", NR_FPGA_CHANNELS: " << T::NR_FPGA_CHANNELS
              << ", NR_RECEIVERS: " << T::NR_RECEIVERS
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

    if constexpr (T::NR_FINE_CHANNELS > 1) {
      d_samples_pre_align.resize(num_buffers);
      d_samples_aligned.resize(num_buffers);
      d_samples_reordered.resize(num_buffers);
      d_channelizer_input.resize(num_buffers);
      d_channelizer_output.resize(num_buffers);
    } else {
      fft_plan.resize(num_buffers);
      d_cufft_work_area.resize(num_buffers);
    }

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

      if constexpr (T::NR_FINE_CHANNELS > 1) {
        CUDA_CHECK(cudaMalloc((void **)&d_samples_pre_align[i],
                              sizeof(typename T::HalfPacketSamplesType)));
        CUDA_CHECK(
            cudaMalloc((void **)&d_samples_aligned[i],
                       sizeof(typename T::HalfPacketAlignedSamplesType)));
        CUDA_CHECK(
            cudaMalloc((void **)&d_samples_reordered[i],
                       sizeof(typename T::HalfPacketAlignedSamplesType)));
        CUDA_CHECK(cudaMalloc(
            (void **)&d_channelizer_input[i],
            sizeof(typename FineChannelizer<T>::FilterInputType)));
        CUDA_CHECK(cudaMalloc(
            (void **)&d_channelizer_output[i],
            sizeof(typename FineChannelizer<T>::FilterOutputType)));
      }
    }

    // Default (unity) calibration gains, per T::NR_FPGA_CHANNELS -- matches every other
    // pipeline's constructor; previously never allocated/initialized here (a latent bug: an
    // uninitialized d_gains was passed straight into ingest_and_scale's gain-scaling kernel).
    CUDA_CHECK(cudaMalloc((void **)&d_gains, sizeof(typename T::AntennaGains)));
    auto default_gains = get_default_gains<T::NR_FPGA_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));

    last_frame_processed = 0;
    current_buffer = 0;

    if constexpr (T::NR_FINE_CHANNELS > 1) {
      CUDA_CHECK(cudaMalloc((void **)&d_subpacket_delays,
                            sizeof(int) * T::NR_FPGA_SOURCES));
      CUDA_CHECK(
          cudaMemset(d_subpacket_delays, 0, sizeof(int) * T::NR_FPGA_SOURCES));
      {
        constexpr int NR_RECVS = T::NR_FPGA_SOURCES * T::NR_RECEIVERS_PER_PACKET;
        constexpr int NR_PERM = NR_RECVS * (int)T::NR_POLARIZATIONS;
        CUDA_CHECK(cudaMalloc((void **)&d_stream_perm_recv, sizeof(int) * NR_PERM));
        CUDA_CHECK(cudaMalloc((void **)&d_stream_perm_pol, sizeof(int) * NR_PERM));
        std::vector<int> identity_recv(NR_PERM), identity_pol(NR_PERM);
        for (int i = 0; i < NR_RECVS; ++i)
          for (int p = 0; p < (int)T::NR_POLARIZATIONS; ++p) {
            identity_recv[i * T::NR_POLARIZATIONS + p] = i;
            identity_pol[i * T::NR_POLARIZATIONS + p] = p;
          }
        CUDA_CHECK(cudaMemcpy(d_stream_perm_recv, identity_recv.data(),
                              sizeof(int) * NR_PERM, cudaMemcpyDefault));
        CUDA_CHECK(cudaMemcpy(d_stream_perm_pol, identity_pol.data(),
                              sizeof(int) * NR_PERM, cudaMemcpyDefault));
      }

      CUdevice cu_device;
      cuDeviceGet(&cu_device, 0);
      channelizer_ = std::make_unique<FineChannelizer<T>>(cu_device);
    }

    cudaDeviceSynchronize();
    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modePacketPreAlign, "prealign");
    tensor_16.addTensor(modeCUFFTInput, "cufftInput");

    tensor_16.addPermutation("packet", "prealign", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToPreAlign");
    tensor_16.addPermutation("packet", "cufftInput", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToCUFFTInput");

    if constexpr (T::NR_FINE_CHANNELS == 1) {
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
    for (auto p : d_samples_pre_align) {
      cudaFree(p);
    }
    for (auto p : d_samples_aligned) {
      cudaFree(p);
    }
    for (auto p : d_samples_reordered) {
      cudaFree(p);
    }
    for (auto p : d_channelizer_input) {
      cudaFree(p);
    }
    for (auto p : d_channelizer_output) {
      cudaFree(p);
    }
    if (d_gains)
      cudaFree(d_gains);
    if (d_subpacket_delays)
      cudaFree(d_subpacket_delays);
    if (d_stream_perm_recv)
      cudaFree(d_stream_perm_recv);
    if (d_stream_perm_pol)
      cudaFree(d_stream_perm_pol);
  };

  void dump_visibilities(const uint64_t end_seq_num = 0) override {
    // nothing to do.
  }
};
