#pragma once

// Correlation-only pipeline: full ingest -> per-FPGA delay correction ->
// receiver/pol reordering -> pre-correlation fine channelization -> TCC
// correlation -> visibilities, with no beamforming, no eigendecomposition/RFI
// mitigation, and no spectral (FFT) output. Modeled on LambdaGPUPipeline's
// ingest/align/channelize/correlate sequence (not LambdaCorrBeamOnlyGPUPipeline,
// whose disabled path is a fused benchmark-only fast path with no delay/reorder
// stage), with everything downstream of correlation removed.
template <typename T> class LambdaStarweavePipeline : public GPUPipeline {

  static_assert((T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET) %
                        T::NR_FINE_CHANNELS ==
                    0,
                "NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET must divide evenly "
                "by NR_FINE_CHANNELS");

private:
  int num_buffers;

  static constexpr int NR_TIMES_PER_BLOCK = 128 / 16; // fp16 pipeline
  static constexpr int NR_BLOCKS_FOR_CORRELATION =
      (T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET /
       T::NR_FINE_CHANNELS) /
      NR_TIMES_PER_BLOCK;
  static constexpr int NR_BASELINES =
      T::NR_PADDED_RECEIVERS * (T::NR_PADDED_RECEIVERS + 1) / 2;
  static constexpr int NR_UNPADDED_BASELINES =
      T::NR_RECEIVERS * (T::NR_RECEIVERS + 1) / 2;
  static constexpr int COMPLEX = 2;
  const int NR_CORRELATED_BLOCKS_TO_ACCUMULATE;

  inline static const __half alpha = __float2half(1.0f);

  using CorrelatorInput =
      __half[T::NR_CHANNELS][NR_BLOCKS_FOR_CORRELATION][T::NR_PADDED_RECEIVERS]
            [T::NR_POLARIZATIONS][NR_TIMES_PER_BLOCK][COMPLEX];

  using CorrelatorOutput =
      float[T::NR_CHANNELS][NR_BASELINES][T::NR_POLARIZATIONS]
           [T::NR_POLARIZATIONS][COMPLEX];

  using TrimmedVisibilities =
      std::complex<float>[T::NR_CHANNELS][NR_UNPADDED_BASELINES]
                         [T::NR_POLARIZATIONS][T::NR_POLARIZATIONS];

  // C (capital) = NR_FPGA_CHANNELS, raw pre-channelization channel count -- both tensors
  // registered here are pre-channelization (packet -> pre-align bridge), same convention as
  // LambdaGPUPipeline's identically-named modes.
  // f = fpga, g = packets for correlation + 2 (pre-alignment padding), n = receivers per packet,
  // p = polarization, u = time steps per packet, z = complex.
  inline static const std::vector<int> modePacket{'C', 'g', 'f', 'u', 'n', 'p', 'z'};
  inline static const std::vector<int> modePacketPreAlign{'f', 'g', 'u', 'C', 'n', 'p', 'z'};

  inline static const std::unordered_map<int, int64_t> extent = {
      {'C', T::NR_FPGA_CHANNELS},
      {'f', T::NR_FPGA_SOURCES},
      {'g', T::NR_PACKETS_FOR_CORRELATION + 2},
      {'n', T::NR_RECEIVERS_PER_PACKET},
      {'p', T::NR_POLARIZATIONS},
      {'u', T::NR_TIME_STEPS_PER_PACKET},
      {'z', 2},
  };

  CutensorSetup tensor_16;

  int current_buffer;
  std::atomic<int> num_correlation_units_integrated;

  tcc::Correlator correlator;

  struct PipelineResources {
    cudaStream_t stream = nullptr;
    cudaStream_t host_stream = nullptr;
    cudaEvent_t ingest_copy_done = nullptr;

    DevicePtr<typename T::InputPacketSamplesType> samples_entry;
    DevicePtr<typename T::PacketScalesType> scales;
    DevicePtr<typename T::HalfPacketSamplesType> samples_half, samples_pre_align;
    DevicePtr<typename T::HalfPacketAlignedSamplesType> samples_aligned,
        samples_reordered;

    // Fine-channelization scratch (only used when T::NR_FINE_CHANNELS > 1; allocated
    // unconditionally for simplicity, matching LambdaGPUPipeline's convention).
    DevicePtr<typename FineChannelizer<T>::FilterInputType> channelizer_input;
    DevicePtr<typename FineChannelizer<T>::FilterOutputType> channelizer_output;

    DevicePtr<CorrelatorInput> correlator_input;
    DevicePtr<CorrelatorOutput> correlator_output;
    DevicePtr<TrimmedVisibilities> visibilities_trimmed;

    // Two static mid-pipeline sections captured as CUDA graphs (see capture_graph) --
    // alignment (permute + delays + reorder) and channelize+correlate+trim. Unlike
    // LambdaGPUPipeline there's no eigen/beamform section, so accumulate_visibilities stays
    // eager (cheap, single kernel call) rather than needing a third graph.
    cudaGraphExec_t graph_align = nullptr;
    cudaGraphExec_t graph_corr = nullptr;

    // Recorded on `stream` right after this buffer's accumulate_visibilities call.
    // dump_visibilities waits on this from every buffer before reading the accumulator.
    cudaEvent_t accumulate_done = nullptr;

    PipelineResources() = default;

    explicit PipelineResources(CUdevice /*cu_device*/)
        : samples_entry(make_device_ptr<typename T::InputPacketSamplesType>()),
          scales(make_device_ptr<typename T::PacketScalesType>()),
          samples_half(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_pre_align(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_aligned(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          samples_reordered(
              make_device_ptr<typename T::HalfPacketAlignedSamplesType>()),
          channelizer_input(
              make_device_ptr<typename FineChannelizer<T>::FilterInputType>()),
          channelizer_output(
              make_device_ptr<typename FineChannelizer<T>::FilterOutputType>()),
          correlator_input(make_device_ptr<CorrelatorInput>()),
          correlator_output(make_device_ptr<CorrelatorOutput>()),
          visibilities_trimmed(make_device_ptr<TrimmedVisibilities>()) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUDA_CHECK(
          cudaStreamCreateWithFlags(&host_stream, cudaStreamNonBlocking));
      CUDA_CHECK(cudaEventCreateWithFlags(&ingest_copy_done,
                                          cudaEventDisableTiming));
    }

    ~PipelineResources() {
      if (graph_align)
        cudaGraphExecDestroy(graph_align);
      if (graph_corr)
        cudaGraphExecDestroy(graph_corr);
      if (accumulate_done)
        cudaEventDestroy(accumulate_done);
      if (ingest_copy_done)
        cudaEventDestroy(ingest_copy_done);
      if (stream)
        cudaStreamDestroy(stream);
      if (host_stream)
        cudaStreamDestroy(host_stream);
    }

    PipelineResources(PipelineResources &&other) noexcept
        : stream(other.stream), host_stream(other.host_stream),
          ingest_copy_done(other.ingest_copy_done),
          samples_entry(std::move(other.samples_entry)),
          scales(std::move(other.scales)),
          samples_half(std::move(other.samples_half)),
          samples_pre_align(std::move(other.samples_pre_align)),
          samples_aligned(std::move(other.samples_aligned)),
          samples_reordered(std::move(other.samples_reordered)),
          channelizer_input(std::move(other.channelizer_input)),
          channelizer_output(std::move(other.channelizer_output)),
          correlator_input(std::move(other.correlator_input)),
          correlator_output(std::move(other.correlator_output)),
          visibilities_trimmed(std::move(other.visibilities_trimmed)),
          graph_align(other.graph_align), graph_corr(other.graph_corr),
          accumulate_done(other.accumulate_done) {
      other.stream = nullptr;
      other.host_stream = nullptr;
      other.ingest_copy_done = nullptr;
      other.graph_align = nullptr;
      other.graph_corr = nullptr;
      other.accumulate_done = nullptr;
    }

    PipelineResources &operator=(PipelineResources &&other) noexcept {
      if (this != &other) {
        this->~PipelineResources();
        new (this) PipelineResources(std::move(other));
      }
      return *this;
    }

    PipelineResources(const PipelineResources &) = delete;
    PipelineResources &operator=(const PipelineResources &) = delete;
  };

  TrimmedVisibilities *d_visibilities_accumulator;

  // Recorded on buffers[0].stream after dump_visibilities resets
  // d_visibilities_accumulator. Every buffer's next accumulate_visibilities waits on this,
  // so the reset can never race with an in-flight accumulation.
  cudaEvent_t visibilities_reset_done = nullptr;

  typename T::AntennaGains *d_gains;
  std::unique_ptr<FineChannelizer<T>> channelizer_;
  std::vector<PipelineResources> buffers;
  // One H2D completion event per host packet-assembly buffer. A host buffer cannot be handed
  // back to ProcessorState (and therefore cannot be submitted again) until its own event has
  // completed, so these events are never re-recorded while an earlier wait is outstanding.
  // The old one-event-per-GPU-stream scheme violated that ownership relationship when several
  // host buffers were queued on the same stream and eventually stopped running release callbacks.
  std::vector<cudaEvent_t> host_buffer_copy_done;
  std::unique_ptr<std::atomic<bool>[]> host_buffer_release_pending;
  size_t host_buffer_release_count = 0;
  std::atomic<bool> buffer_reclaimer_running{false};
  std::thread buffer_reclaimer_thread;
  int *d_subpacket_delays;
  int *d_stream_perm_recv = nullptr; // [NR_RECEIVERS*NR_POL] canonical->src flat recv; identity by default
  int *d_stream_perm_pol = nullptr;  // [NR_RECEIVERS*NR_POL] canonical->src hw pol slot; identity by default
  uint64_t visibilities_start_seq_num = 0;
  uint64_t visibilities_end_seq_num = 0;
  bool visibilities_have_seq = false;
  static constexpr int visibilities_total_packets_per_block =
      T::NR_FPGA_CHANNELS * T::NR_PACKETS_FOR_CORRELATION * T::NR_FPGA_SOURCES;
  int visibilities_missing_packets;

public:
  void *gpu_landing_samples_ptr(int buffer_index) override {
    return buffers[buffer_index].samples_entry.get();
  }
  void *gpu_landing_scales_ptr(int buffer_index) override {
    return buffers[buffer_index].scales.get();
  }

  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {
    auto &b = buffers[current_buffer];

    if (!dummy_run) {
      if (!visibilities_have_seq) {
        visibilities_start_seq_num = packet_data->start_seq_id;
        visibilities_have_seq = true;
      }
      visibilities_end_seq_num = packet_data->end_seq_id;
      visibilities_missing_packets += packet_data->get_num_missing_packets();
    }

    cudaEvent_t copy_done_event = b.ingest_copy_done;
    if (!dummy_run) {
      if (host_buffer_copy_done.empty()) {
        const size_t count = this->state_->input_buffer_count();
        if (count == 0) {
          throw std::logic_error(
              "ProcessorState did not report its host input-buffer count");
        }
        host_buffer_copy_done.resize(count, nullptr);
        for (auto &event : host_buffer_copy_done) {
          CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        }
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
                // Clear ownership before release_buffer() publishes the buffer. The feeder may
                // legitimately submit that buffer again as soon as release_buffer() returns.
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
        INFO_LOG("Starweave ingest created {} per-host-buffer H2D completion events",
                 count);
      }
      const int host_buffer_index = packet_data->buffer_index;
      if (host_buffer_index < 0 ||
          static_cast<size_t>(host_buffer_index) >= host_buffer_copy_done.size()) {
        throw std::out_of_range("packet_data host buffer index is out of range");
      }
      if (host_buffer_release_pending[host_buffer_index].load(
              std::memory_order_acquire)) {
        throw std::logic_error(
            "host packet buffer was resubmitted before its H2D completion");
      }
      copy_done_event = host_buffer_copy_done[host_buffer_index];
    }
    LambdaPipelineIngest<T>::ingest_and_scale(
        this->state_, packet_data, b.stream, b.host_stream,
        b.samples_entry.get(), b.scales.get(), d_gains, b.samples_half.get(),
        dummy_run, copy_done_event, dummy_run);
    if (!dummy_run) {
      host_buffer_release_pending[packet_data->buffer_index].store(
          true, std::memory_order_release);
    }

    if (b.graph_align != nullptr) {
      CUDA_CHECK(cudaGraphLaunch(b.graph_align, b.stream));
    } else {
      enqueue_alignment(b);
    }

    if (b.graph_corr != nullptr) {
      CUDA_CHECK(cudaGraphLaunch(b.graph_corr, b.stream));
    } else {
      enqueue_corr(b);
    }

    // GPU-side wait (not a host-blocking sync): make this buffer's accumulate wait for
    // dump_visibilities' last reset of d_visibilities_accumulator to land first. No-op
    // until the first dump happens (an event that's never been recorded is treated as
    // already satisfied).
    cudaStreamWaitEvent(b.stream, visibilities_reset_done, 0);
    accumulate_visibilities(
        (float *)b.visibilities_trimmed.get(), (float *)d_visibilities_accumulator,
        2 * NR_UNPADDED_BASELINES * T::NR_POLARIZATIONS * T::NR_POLARIZATIONS *
            T::NR_CHANNELS,
        b.stream);
    cudaEventRecord(b.accumulate_done, b.stream);

    if (output_ != nullptr && !dummy_run) {
      num_correlation_units_integrated += 1;
      if (num_correlation_units_integrated >= NR_CORRELATED_BLOCKS_TO_ACCUMULATE) {
        dump_visibilities();
      }
    }

    if (!dummy_run) {
      current_buffer = (current_buffer + 1) % num_buffers;
    }

  }

  // Part 1: permute + apply integer delays -> samples_aligned, then reorder receivers
  // into canonical antenna-ID order -> samples_reordered. Identical to
  // LambdaGPUPipeline::enqueue_alignment.
  void enqueue_alignment(PipelineResources &b) {
    tensor_16.runPermutation("packetToPreAlign", alpha,
                             (__half *)b.samples_half.get(),
                             (__half *)b.samples_pre_align.get(), b.stream);

    apply_delays_launch((__half *)b.samples_pre_align.get(),
                        (__half *)b.samples_aligned.get(), d_subpacket_delays,
                        T::NR_RECEIVERS_PER_PACKET, T::NR_FPGA_SOURCES,
                        T::NR_PACKETS_FOR_CORRELATION, T::NR_POLARIZATIONS,
                        T::NR_FPGA_CHANNELS, T::NR_TIME_STEPS_PER_PACKET, b.stream);

    reorder_streams_launch<T::NR_FPGA_SOURCES, T::NR_PACKETS_FOR_CORRELATION,
                           T::NR_TIME_STEPS_PER_PACKET, T::NR_FPGA_CHANNELS,
                           T::NR_RECEIVERS_PER_PACKET, T::NR_POLARIZATIONS>(
        (__half *)b.samples_aligned.get(), (__half *)b.samples_reordered.get(),
        d_stream_perm_recv, d_stream_perm_pol, b.stream);
  }

  // Part 2: corr_input reformat (fine-channelization, or the plain reshape when disabled) +
  // TCC + trim. No decomp/eigen permutation -- visibilities output doesn't need the
  // unpacked square form cuSOLVER would need.
  void enqueue_corr(PipelineResources &b) {
    if constexpr (T::NR_FINE_CHANNELS > 1) {
      reorder_to_filter_input<T::NR_FPGA_CHANNELS, T::NR_POLARIZATIONS, T::NR_RECEIVERS,
                              T::NR_RECEIVERS_PER_PACKET, T::NR_TIME_STEPS_PER_PACKET,
                              T::NR_PACKETS_FOR_CORRELATION, FineChannelizer<T>::NR_TAPS,
                              T::NR_FINE_CHANNELS>(
          (__half *)b.samples_reordered.get(),
          (float2 *)b.channelizer_input.get(), b.stream);

      channelizer_->launchAsync(b.stream, b.channelizer_input.get(),
                                b.channelizer_output.get());

      channelizer_output_to_corr_input<T::NR_FPGA_CHANNELS, T::NR_FINE_CHANNELS,
                                       T::NR_FINE_CHANNEL_EDGE_TRIM,
                                       T::NR_POLARIZATIONS, T::NR_RECEIVERS,
                                       T::NR_PADDED_RECEIVERS,
                                       NR_BLOCKS_FOR_CORRELATION, NR_TIMES_PER_BLOCK>(
          (const __half2 *)b.channelizer_output.get(),
          (__half *)b.correlator_input.get(), b.stream,
          channelizer_->deripple_gains());
    } else {
      aligned_to_corr_input<T::NR_CHANNELS, T::NR_POLARIZATIONS, T::NR_RECEIVERS,
                            T::NR_RECEIVERS_PER_PACKET,
                            T::NR_TIME_STEPS_PER_PACKET,
                            T::NR_PACKETS_FOR_CORRELATION,
                            T::NR_PADDED_RECEIVERS, NR_TIMES_PER_BLOCK>(
          (__half *)b.samples_reordered.get(), (__half *)b.correlator_input.get(),
          b.stream);
    }

    correlator.launchAsync((CUstream)b.stream,
                           (CUdeviceptr)b.correlator_output.get(),
                           (CUdeviceptr)b.correlator_input.get());

    corr_to_trimmed((float *)b.correlator_output.get(),
                    (float *)b.visibilities_trimmed.get(), T::NR_CHANNELS,
                    NR_BASELINES, NR_UNPADDED_BASELINES,
                    T::NR_POLARIZATIONS * T::NR_POLARIZATIONS * 2, b.stream);
  }

  // Capture one section into an instantiated graph. Raw CUDA calls (not CUDA_CHECK,
  // which exits the process) so a capture-incompatible op degrades to eager execution.
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

  LambdaStarweavePipeline(const int num_buffers,
                         int nr_blocks_to_accumulate = T::NR_CORRELATED_BLOCKS_TO_ACCUMULATE)
      : num_buffers(num_buffers),
        NR_CORRELATED_BLOCKS_TO_ACCUMULATE(nr_blocks_to_accumulate),
        tensor_16(extent, CUTENSOR_R_16F, 128),
        correlator(cu::Device(0), tcc::Format::fp16, T::NR_PADDED_RECEIVERS,
                   T::NR_CHANNELS,
                   NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK,
                   T::NR_POLARIZATIONS, std::nullopt,
                   T::NR_PADDED_RECEIVERS_PER_BLOCK,
                   TCC_THREAD_BLOCKS_PER_SM) {
    std::cout << "LambdaStarweavePipeline instantiated with NR_CHANNELS: "
              << T::NR_CHANNELS << ", NR_RECEIVERS: " << T::NR_PADDED_RECEIVERS
              << ", NR_POLARIZATIONS: " << T::NR_POLARIZATIONS
              << ", NR_SAMPLES_PER_CHANNEL: "
              << NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK << std::endl;

    CUDA_CHECK(cudaMalloc((void **)&d_visibilities_accumulator,
                          sizeof(TrimmedVisibilities)));
    // accumulate_visibilities uses atomicAdd, including during constructor warmup.
    // cudaMalloc does not promise zero-filled memory.
    CUDA_CHECK(cudaMemset(d_visibilities_accumulator, 0,
                          sizeof(TrimmedVisibilities)));
    CUDA_CHECK(cudaEventCreateWithFlags(&visibilities_reset_done,
                                        cudaEventDisableTiming));

    CUDA_CHECK(cudaMalloc((void **)&d_subpacket_delays,
                          sizeof(int) * T::NR_FPGA_SOURCES));
    CUDA_CHECK(
        cudaMemset(d_subpacket_delays, 0, sizeof(int) * T::NR_FPGA_SOURCES));

    CUDA_CHECK(cudaMalloc((void **)&d_gains, sizeof(typename T::AntennaGains)));
    auto default_gains = get_default_gains<T::NR_FPGA_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));

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

    num_correlation_units_integrated = 0;
    current_buffer = 0;
    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);

    if constexpr (T::NR_FINE_CHANNELS > 1) {
      channelizer_ = std::make_unique<FineChannelizer<T>>(cu_device);
    }

    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modePacketPreAlign, "prealign");
    tensor_16.addPermutation("packet", "prealign", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToPreAlign");

    buffers.reserve(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(cu_device);
      auto &b = buffers.back();
      CUDA_CHECK(
          cudaEventCreateWithFlags(&b.accumulate_done, cudaEventDisableTiming));
      CUDA_CHECK(cudaMemsetAsync(b.correlator_input.get(), 0,
                                 sizeof(CorrelatorInput), b.stream));
    }

    cudaDeviceSynchronize();

    // Warm up the pipeline before attempting graph capture -- see LambdaGPUPipeline's
    // identical warmup for why (NVRTC JIT / lazy library init isn't safe during
    // cudaStreamBeginCapture).
    typename T::PacketFinalDataType warmup_packet;
    std::memset(warmup_packet.samples, 0,
                warmup_packet.get_samples_elements_size());
    std::memset(warmup_packet.scales, 0,
                warmup_packet.get_scales_element_size());
    std::memset(warmup_packet.arrivals, 0, warmup_packet.get_arrivals_size());
    execute_pipeline(&warmup_packet, true);
    cudaDeviceSynchronize();

    {
      bool all_ok = true;
      for (auto &b : buffers) {
        if (!capture_graph(
                b, [this](PipelineResources &r) { enqueue_alignment(r); },
                b.graph_align) ||
            !capture_graph(
                b, [this](PipelineResources &r) { enqueue_corr(r); },
                b.graph_corr)) {
          all_ok = false;
          break;
        }
      }
      if (!all_ok) {
        for (auto &b : buffers) {
          if (b.graph_align) {
            cudaGraphExecDestroy(b.graph_align);
            b.graph_align = nullptr;
          }
          if (b.graph_corr) {
            cudaGraphExecDestroy(b.graph_corr);
            b.graph_corr = nullptr;
          }
        }
        WARN_LOG("CUDA graph capture failed — LambdaStarweavePipeline will run eagerly");
      } else {
        INFO_LOG("CUDA graphs captured for {} LambdaStarweavePipeline buffers",
                 buffers.size());
        execute_pipeline(&warmup_packet, true);
      }
      cudaDeviceSynchronize();
    }

    visibilities_have_seq = false;
    visibilities_missing_packets = 0;
  };

  ~LambdaStarweavePipeline() {
    buffer_reclaimer_running.store(false, std::memory_order_release);
    if (buffer_reclaimer_thread.joinable())
      buffer_reclaimer_thread.join();
    // All host-stream waits/callbacks must retire before their per-host-buffer events are
    // destroyed. cudaStreamSynchronize also makes shutdown errors visible to CUDA_CHECK.
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
    if (visibilities_reset_done)
      cudaEventDestroy(visibilities_reset_done);
    if (d_visibilities_accumulator)
      cudaFree(d_visibilities_accumulator);
    if (d_gains)
      cudaFree(d_gains);
    if (d_subpacket_delays)
      cudaFree(d_subpacket_delays);
    if (d_stream_perm_recv)
      cudaFree(d_stream_perm_recv);
    if (d_stream_perm_pol)
      cudaFree(d_stream_perm_pol);
  };

  void set_subpacket_delays(int *delays_subpacket) override {
    subpacket_delays_ = delays_subpacket;
    CUDA_CHECK(cudaMemcpy(d_subpacket_delays, subpacket_delays_,
                          sizeof(int) * T::NR_FPGA_SOURCES, cudaMemcpyDefault));
  }

  void set_antenna_gains(std::complex<float> *gains) override {
    std::cout << "setting antenna gains on LambdaStarweavePipeline...\n";
    gains_ = gains;
    CUDA_CHECK(cudaMemcpy(d_gains, gains, sizeof(typename T::AntennaGains),
                          cudaMemcpyDefault));
    cudaDeviceSynchronize();
    std::cout << "gains uploaded successfully...\n";
  }

  // Upload stream reorder permutation tables. See LambdaGPUPipeline::set_stream_permutation
  // for the exact convention (recv_perm[canonical_idx] = hw flat receiver, -1 = unused;
  // pol_perm[canonical_idx] = src pol index carrying X).
  void set_stream_permutation(const std::vector<int> &recv_perm,
                              const std::vector<int> &pol_perm) override {
    constexpr int NR_RECVS = T::NR_FPGA_SOURCES * T::NR_RECEIVERS_PER_PACKET;
    constexpr int NR_PERM = NR_RECVS * (int)T::NR_POLARIZATIONS;
    if ((int)recv_perm.size() != NR_PERM || (int)pol_perm.size() != NR_PERM)
      throw std::runtime_error(
          "set_stream_permutation: size must equal NR_FPGA_SOURCES * NR_RECEIVERS_PER_PACKET * NR_POLARIZATIONS");
    CUDA_CHECK(cudaMemcpy(d_stream_perm_recv, recv_perm.data(),
                          sizeof(int) * NR_PERM, cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(d_stream_perm_pol, pol_perm.data(),
                          sizeof(int) * NR_PERM, cudaMemcpyDefault));
    std::cout << "Stream reorder permutation uploaded.\n";
  }

  void dump_visibilities(const uint64_t end_seq_num = 0) override {
    INFO_LOG("Dumping correlations to host...");
    int current_num_integrated_units_processed = num_correlation_units_integrated;
    INFO_LOG("Current num integrated units processed is {}",
             current_num_integrated_units_processed);
    if (current_num_integrated_units_processed == 0) {
      return;
    }
    if (!visibilities_have_seq) {
      throw std::logic_error("Cannot dump visibilities without a real packet sequence");
    }
    for (auto &buf : buffers) {
      cudaStreamWaitEvent(buffers[0].stream, buf.accumulate_done, 0);
    }
    const int visibilities_total_packets =
        current_num_integrated_units_processed *
        visibilities_total_packets_per_block;
    size_t block_num = output_->register_visibilities_block(
        visibilities_start_seq_num,
        end_seq_num == 0 ? visibilities_end_seq_num : end_seq_num,
        visibilities_missing_packets,
        visibilities_total_packets);
    visibilities_have_seq = false;
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
    cudaEventRecord(visibilities_reset_done, buffers[0].stream);
  };
};
