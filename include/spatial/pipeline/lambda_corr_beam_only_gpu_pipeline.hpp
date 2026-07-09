#pragma once

template <typename T> class LambdaCorrBeamOnlyGPUPipeline : public GPUPipeline {

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
  static constexpr int NR_UNPADDED_BASELINES =
      T::NR_RECEIVERS * (T::NR_RECEIVERS + 1) / 2;
  static constexpr int NR_TIME_STEPS_FOR_CORRELATION =
      T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET;
  static constexpr int COMPLEX = 2;
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

  typename T::AntennaGains *d_gains;
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
  inline static const std::vector<int> modePacketPadding{'f', 'n', 'c', 'o',
                                                         'u', 'p', 'z'};
  inline static const std::vector<int> modePacketPadded{'d', 'c', 'b',
                                                        't', 'p', 'z'};
  inline static const std::vector<int> modeCorrelatorInput{'c', 'b', 'd',
                                                           'p', 't', 'z'};
  inline static const std::vector<int> modePlanar{'c', 'p', 'z', 'f',
                                                  'n', 'o', 'u'};
  inline static const std::vector<int> modeCUFFTInput{'c', 'p', 'f', 'n',
                                                      'o', 'u', 'z'};
  // We need the planar samples matrix to be in column-major memory layout
  // which is equivalent to transposing time and receiver structure here.
  // We also squash the b,t axes into s = block * time
  // CCGLIB requires that we have BLOCK x COMPLEX x COL x ROW structure.
  inline static const std::vector<int> modePlanarCons = {'c', 'p', 'z',
                                                         'f', 'n', 's'};
  inline static const std::vector<int> modePlanarColMajCons = {'c', 'p', 'z',
                                                               's', 'f', 'n'};
  // Fused: packet → col-maj-cons in one step (replaces packetToPlanar +
  // consToColMajCons). o,u adjacent in output with u inner gives stride(s)=
  // stride(u), same layout as ['c','p','z','s','f','n'] with s=o*u.
  inline static const std::vector<int> modePacketDirectColMajCons = {
      'c', 'p', 'z', 'o', 'u', 'f', 'n'};
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

  std::vector<std::unique_ptr<ccglib::pipeline::Pipeline>> gemm_handles;

  tcc::Correlator correlator;

  std::vector<typename T::InputPacketSamplesType *> d_samples_entry;

  std::vector<typename T::HalfPacketSamplesType *> d_samples_half;
  std::vector<CorrelatorInput *> d_correlator_input;
  std::vector<CorrelatorOutput *> d_correlator_output;

  TrimmedVisibilities *d_visibilities_accumulator;

  // Captured replay of the static per-buffer permutation/correlator/gemm
  // sequence (see enqueue_main); a nullptr entry falls back to the eager
  // enqueue_main(i) call.
  std::vector<cudaGraphExec_t> graph_main;

  // Recorded on streams[i] right after enqueue_main's
  // accumulate_visibilities lands. dump_visibilities waits on all of these
  // before reading d_visibilities_accumulator, instead of a
  // cudaDeviceSynchronize().
  std::vector<cudaEvent_t> accumulate_done;

  // Recorded on streams[0] after dump_visibilities resets
  // d_visibilities_accumulator. Each buffer's enqueue_main waits on this
  // before its next accumulate_visibilities, so the reset can never race
  // with an in-flight accumulation -- without a cudaDeviceSynchronize().
  cudaEvent_t visibilities_reset_done = nullptr;

  std::vector<BeamformerOutput *> d_beamformer_output;
  std::vector<HalfBeamformerOutput *> d_beamformer_data_output_half;
  std::vector<__half *> d_samples_consolidated_col_maj, d_weights,
      d_weights_permuted;
  std::vector<typename T::PacketScalesType *> d_scales;

  BeamWeights *h_weights;
  // See the comment on this member in LambdaGPUPipeline (pipeline.hpp above)
  // -- same helper, same inert-when-unsteered contract. Note this pipeline
  // keeps its weights in d_weights[buffer_index]/streams[buffer_index]
  // (rather than buffers[i].weights/.stream); register_buffer() doesn't care
  // -- it just needs each buffer's device pointer and stream.
  BeamSteering<T> beam_steering_;

  int visibilities_start_seq_num;
  int visibilities_end_seq_num;
  static constexpr int visibilities_total_packets_per_block =
      T::NR_CHANNELS * T::NR_PACKETS_FOR_CORRELATION * T::NR_FPGA_SOURCES;
  int visibilities_missing_packets;

public:
  static constexpr size_t NR_BENCHMARKING_RUNS = 100;
  size_t benchmark_runs_done = 0;
  cudaEvent_t start_run[NR_BENCHMARKING_RUNS], stop_run[NR_BENCHMARKING_RUNS];
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {

    const size_t start_seq_num = packet_data->start_seq_id;
    const size_t end_seq_num = packet_data->end_seq_id;
    if (visibilities_start_seq_num == -1) {
      visibilities_start_seq_num = packet_data->start_seq_id;
    }
    visibilities_missing_packets += packet_data->get_num_missing_packets();

    // Re-steer tracked beams if due -- inert no-op when steering is disabled
    // (no --targets-filename). A due refresh enqueues the new weights onto
    // *every* buffer's stream in this one call (the d_weights[i]/streams[i]
    // pairs registered in the constructor below), so all buffers always run
    // with identical weights. Must run here, before anything below reads
    // d_weights[current_buffer], and only from this single-threaded
    // pipeline_feeder context -- see the BeamSteering<T> comment block
    // (pipeline.hpp above) for why that ordering is what makes this safe
    // without extra synchronization.
    beam_steering_.maybe_refresh();

    // Record GPU start event
    cudaEventRecord(start_run[benchmark_runs_done], streams[current_buffer]);

    // dummy_run must be forwarded: on the constructor's warmup run state_
    // is not set yet and packet_data->buffer_index is meaningless, so the
    // release_buffer host func has to skip the release (a hardcoded false
    // here made the warmup segfault in release_buffer_host_func).
    LambdaPipelineIngest<T>::ingest_and_scale(
        this->state_, packet_data, streams[current_buffer],
        streams[current_buffer], d_samples_entry[current_buffer],
        d_scales[current_buffer], d_gains, d_samples_half[current_buffer],
        dummy_run);

    // enqueue_main's first op (accumulate_visibilities) adds into
    // d_visibilities_accumulator. Wait for dump_visibilities' last reset of
    // that accumulator (on streams[0]) to land first -- a GPU-side wait,
    // not a host-blocking sync. No-op until the first dump happens.
    cudaStreamWaitEvent(streams[current_buffer], visibilities_reset_done, 0);

    // Static mid-pipeline section: a single graph launch replaces ~15
    // individual launches when capture succeeded at construction (see
    // enqueue_main / capture_graph).
    if (graph_main[current_buffer] != nullptr) {
      CUDA_CHECK(
          cudaGraphLaunch(graph_main[current_buffer], streams[current_buffer]));
    } else {
      enqueue_main(current_buffer);
    }

    // Lets dump_visibilities (on streams[0]) wait for this buffer's
    // accumulate_visibilities to land before reading
    // d_visibilities_accumulator, without a cudaDeviceSynchronize().
    cudaEventRecord(accumulate_done[current_buffer], streams[current_buffer]);

    cudaEventRecord(stop_run[benchmark_runs_done], streams[current_buffer]);

    // Output handling
    if (output_ != nullptr && !dummy_run) {
      size_t block_num =
          output_->register_beam_data_block(start_seq_num, end_seq_num);
      // size_t::max means no beam writer is attached (e.g. observe.cu's
      // beam_writer = nullptr): the landing pointers would be nullptr, so
      // skip the beam/arrivals copies entirely.
      if (block_num != std::numeric_limits<size_t>::max()) {
        void *landing_pointer =
            output_->get_beam_data_landing_pointer(block_num);
        cudaMemcpyAsync(landing_pointer,
                        d_beamformer_data_output_half[current_buffer],
                        sizeof(HalfBeamformerOutput), cudaMemcpyDefault,
                        streams[current_buffer]);
        auto *output_ctx = new OutputTransferCompleteContext{
            .output = this->output_, .block_index = block_num};
        cudaLaunchHostFunc(streams[current_buffer],
                           output_transfer_complete_host_func, output_ctx);

        // memcpy arrivals
        bool *arrivals_output_pointer =
            (bool *)output_->get_arrivals_data_landing_pointer(block_num);
        std::memcpy(arrivals_output_pointer, packet_data->get_arrivals_ptr(),
                    packet_data->get_arrivals_size());
        output_->register_arrivals_transfer_complete(block_num);
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

  // ---- Static mid-pipeline section -------------------------------------
  // Everything here operates on device pointers that are fixed for the
  // lifetime of buffer i, which is what allows it to be captured into a CUDA
  // graph (capture_graph) and replayed as a single launch instead of ~15
  // individual launches. Any op added here must use only per-buffer device
  // pointers -- no per-run host pointers -- to keep the capture valid.
  void enqueue_main(int i) {
    packet_to_corr_input<T::NR_CHANNELS, T::NR_POLARIZATIONS, T::NR_RECEIVERS,
                         T::NR_RECEIVERS_PER_PACKET,
                         T::NR_TIME_STEPS_PER_PACKET,
                         T::NR_PACKETS_FOR_CORRELATION,
                         T::NR_PADDED_RECEIVERS, NR_TIMES_PER_BLOCK>(
        (__half *)d_samples_half[i], (__half *)d_correlator_input[i],
        streams[i]);

    correlator.launchAsync((CUstream)streams[i],
                           (CUdeviceptr)d_correlator_output[i],
                           (CUdeviceptr)d_correlator_input[i]);

    // Fused: visCorrToBaseline + D2D trim + visBaselineTrimmedToTrimmed +
    // accumulate_visibilities in one kernel pass (no intermediate buffers).
    accumulate_visibilities_from_corr(
        (float *)d_correlator_output[i], (float *)d_visibilities_accumulator,
        T::NR_CHANNELS, NR_BASELINES, NR_UNPADDED_BASELINES,
        T::NR_POLARIZATIONS * T::NR_POLARIZATIONS * COMPLEX, streams[i]);

    packet_to_col_maj_cons<T::NR_CHANNELS, T::NR_POLARIZATIONS,
                           T::NR_RECEIVERS, T::NR_RECEIVERS_PER_PACKET,
                           T::NR_TIME_STEPS_PER_PACKET,
                           T::NR_PACKETS_FOR_CORRELATION>(
        (__half *)d_samples_half[i],
        (__half *)d_samples_consolidated_col_maj[i], streams[i]);

    tensor_16.runPermutation("weightsInputToCCGLIB", alpha,
                             (__half *)d_weights[i],
                             (__half *)d_weights_permuted[i], streams[i]);
    (*gemm_handles[i])
        .Run((CUdeviceptr)d_weights_permuted[i],
             (CUdeviceptr)d_samples_consolidated_col_maj[i],
             (CUdeviceptr)d_beamformer_output[i]);

    beam_ccglib_to_half_output<T::NR_CHANNELS, T::NR_POLARIZATIONS,
                               T::NR_BEAMS, NR_TIME_STEPS_FOR_CORRELATION>(
        (float *)d_beamformer_output[i],
        (__half *)d_beamformer_data_output_half[i], streams[i]);
  }

  // Capture one section into an instantiated graph. Raw CUDA calls (not
  // CUDA_CHECK, which exits the process) so a capture-incompatible op
  // degrades to eager execution instead of aborting startup.
  template <typename EnqueueFn>
  bool capture_graph(cudaStream_t stream, EnqueueFn &&enqueue,
                     cudaGraphExec_t &exec_out) {
    cudaGraph_t graph = nullptr;
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) !=
        cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    bool enqueue_ok = true;
    try {
      enqueue();
    } catch (...) {
      // ccglib/cudawrappers throw on capture-incompatible calls.
      enqueue_ok = false;
    }
    const cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
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

  LambdaCorrBeamOnlyGPUPipeline(const int num_buffers,
                                BeamWeightsT<T> *h_weights,
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
        tensor_32(extent, CUTENSOR_R_32F, 128)

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

    streams.resize(2 * num_buffers);
    d_weights.resize(num_buffers);
    d_weights_permuted.resize(num_buffers);
    d_samples_entry.resize(num_buffers);
    d_scales.resize(num_buffers);
    d_samples_half.resize(num_buffers);
    d_samples_consolidated_col_maj.resize(num_buffers);
    d_correlator_input.resize(num_buffers);
    d_correlator_output.resize(num_buffers);
    d_beamformer_output.resize(num_buffers);
    d_beamformer_data_output_half.resize(num_buffers);
    CUDA_CHECK(cudaMalloc((void **)&d_gains, sizeof(typename T::AntennaGains)));
    auto default_gains = get_default_gains<T::NR_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));

    for (auto i = 0; i < num_buffers; ++i) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[num_buffers + i],
                                           cudaStreamNonBlocking));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_entry[i],
                            sizeof(typename T::InputPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_half[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_samples_consolidated_col_maj[i],
                            sizeof(typename T::HalfPacketSamplesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_scales[i],
                            sizeof(typename T::PacketScalesType)));
      CUDA_CHECK(cudaMalloc((void **)&d_weights[i], sizeof(BeamWeights)));
      CUDA_CHECK(
          cudaMalloc((void **)&d_weights_permuted[i], sizeof(BeamWeights)));
      CUDA_CHECK(
          cudaMalloc((void **)&d_correlator_input[i], sizeof(CorrelatorInput)));
      CUDA_CHECK(cudaMemset(d_correlator_input[i], 0, sizeof(CorrelatorInput)));
      CUDA_CHECK(cudaMalloc((void **)&d_correlator_output[i],
                            sizeof(CorrelatorOutput)));
      CUDA_CHECK(cudaMalloc((void **)&d_beamformer_output[i],
                            sizeof(BeamformerOutput)));
      CUDA_CHECK(cudaMalloc((void **)&d_beamformer_data_output_half[i],
                            sizeof(HalfBeamformerOutput)));
    }

    CUDA_CHECK(cudaMalloc((void **)&d_visibilities_accumulator,
                          sizeof(TrimmedVisibilities)));
    CUDA_CHECK(cudaEventCreateWithFlags(&visibilities_reset_done,
                                        cudaEventDisableTiming));
    graph_main.assign(num_buffers, nullptr);
    accumulate_done.resize(num_buffers);
    for (auto i = 0; i < num_buffers; ++i) {
      CUDA_CHECK(cudaEventCreateWithFlags(&accumulate_done[i],
                                          cudaEventDisableTiming));
    }
    last_frame_processed = 0;
    num_integrated_units_processed = 0;
    num_correlation_units_integrated = 0;
    current_buffer = 0;
    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);

    const std::complex<float> alpha_ccglib = {1, 0};
    const std::complex<float> beta_ccglib = {0, 0};
    //    tcc::Format inputFormat = tcc::Format::fp16;
    for (auto i = 0; i < num_buffers; ++i) {
      gemm_handles.emplace_back(std::make_unique<ccglib::pipeline::Pipeline>(
          T::NR_CHANNELS * T::NR_POLARIZATIONS, T::NR_BEAMS,
          NR_TIMES_PER_BLOCK * NR_BLOCKS_FOR_CORRELATION, T::NR_RECEIVERS,
          cu_device, streams[i], ccglib::complex_planar, ccglib::complex_planar,
          ccglib::mma::row_major, ccglib::mma::col_major,
          ccglib::mma::row_major, ccglib::ValueType::float16,
          ccglib::ValueType::float32, ccglib::mma::opt, alpha_ccglib,
          beta_ccglib));
    }

    DEBUG_LOG("Copying weights...");
    for (auto i = 0; i < num_buffers; ++i) {
      cudaMemcpy(d_weights[i], h_weights, sizeof(BeamWeights),
                 cudaMemcpyDefault);
      // d_weights[i] is a __half* but is allocated and copied against
      // sizeof(BeamWeights) (above), exactly like every other pipeline's
      // b.weights -- the reinterpret_cast mirrors that existing convention.
      // Registered before the warmup run below, so the first (always-overdue)
      // maybe_refresh() steers every buffer in one shot.
      beam_steering_.register_buffer(
          reinterpret_cast<BeamWeights *>(d_weights[i]), streams[i]);
    }
    for (auto i = 0; i < NR_BENCHMARKING_RUNS; ++i) {
      cudaEventCreate(&start_run[i]);
      cudaEventCreate(&stop_run[i]);
    }

    cudaDeviceSynchronize();
    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modePlanar, "planar");
    tensor_16.addTensor(modePlanarCons, "planarCons");
    tensor_16.addTensor(modePlanarColMajCons, "planarColMajCons");
    tensor_16.addTensor(modeCUFFTInput, "cufftInput");

    tensor_16.addTensor(modeWeightsInput, "weightsInput");
    tensor_16.addTensor(modeWeightsCCGLIB, "weightsCCGLIB");
    tensor_32.addTensor(modeVisCorr, "visCorr");
    tensor_32.addTensor(modeVisDecomp, "visDecomp");
    tensor_32.addTensor(modeVisCorrBaseline, "visBaseline");
    tensor_32.addTensor(modeVisCorrBaselineTrimmed, "visBaselineTrimmed");
    tensor_32.addTensor(modeVisCorrTrimmed, "visCorrTrimmed");

    tensor_16.addPermutation("packet", "cufftInput", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToCUFFTInput");
    tensor_16.addPermutation("packet", "planar", CUTENSOR_COMPUTE_DESC_16F,
                             "packetToPlanar");
    tensor_16.addPermutation("planarCons", "planarColMajCons",
                             CUTENSOR_COMPUTE_DESC_16F, "consToColMajCons");
    tensor_16.addPermutation("weightsInput", "weightsCCGLIB",
                             CUTENSOR_COMPUTE_DESC_16F, "weightsInputToCCGLIB");
    tensor_32.addPermutation("visCorrTrimmed", "visDecomp",
                             CUTENSOR_COMPUTE_DESC_32F, "visCorrToDecomp");
    tensor_32.addPermutation("visCorr", "visBaseline",
                             CUTENSOR_COMPUTE_DESC_32F, "visCorrToBaseline");
    tensor_32.addPermutation("visBaselineTrimmed", "visCorrTrimmed",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "visBaselineTrimmedToTrimmed");

    // Warm up the pipeline *before* attempting graph capture. This is the
    // first-ever call to most cuTENSOR permutations, the TCC correlator, and
    // the ccglib GEMM. Each of these libraries may do one-time lazy
    // initialization on first use -- NVRTC JIT compilation, module loading,
    // on-disk JIT-cache writes -- which is not safe to perform while
    // cudaStreamBeginCapture is active (can hang or segfault, especially
    // with a cold cache). Running this eagerly first ensures capture below
    // only ever records already-loaded kernels. Because everything is
    // zeroed it should have negligible effect on output.
    typename T::PacketFinalDataType warmup_packet;
    std::memset(warmup_packet.samples, 0,
                warmup_packet.get_samples_elements_size());
    std::memset(warmup_packet.scales, 0,
                warmup_packet.get_scales_element_size());
    std::memset(warmup_packet.arrivals, 0, warmup_packet.get_arrivals_size());
    execute_pipeline(&warmup_packet, true);
    cudaDeviceSynchronize();

    // Capture the static mid-pipeline section of each buffer into a CUDA
    // graph (~15 launches collapse into 1 per run). Any failure falls back
    // to eager execution for all buffers -- functionally identical, just
    // more launch overhead. SPATIAL_DISABLE_CUDA_GRAPH=1 forces the eager
    // path.
    if (std::getenv("SPATIAL_DISABLE_CUDA_GRAPH") == nullptr) {
      bool all_ok = true;
      for (auto i = 0; i < num_buffers; ++i) {
        if (!capture_graph(
                streams[i], [this, i]() { enqueue_main(i); }, graph_main[i])) {
          all_ok = false;
          break;
        }
      }
      if (!all_ok) {
        for (auto &g : graph_main) {
          if (g) {
            cudaGraphExecDestroy(g);
            g = nullptr;
          }
        }
        WARN_LOG("CUDA graph capture failed — pipeline will run eagerly");
      } else {
        INFO_LOG("CUDA graphs captured for {} pipeline buffers", num_buffers);
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
  ~LambdaCorrBeamOnlyGPUPipeline() {
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

    for (auto weight : d_weights) {
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

    for (auto beamformer_output : d_beamformer_output) {
      cudaFree(beamformer_output);
    }
    for (auto beamformer_data_output_half : d_beamformer_data_output_half) {
      cudaFree(beamformer_data_output_half);
    }

    for (auto samples_half : d_samples_half) {
      cudaFree(samples_half);
    }

    for (auto samples_consolidated_col_maj : d_samples_consolidated_col_maj) {
      cudaFree(samples_consolidated_col_maj);
    }

    for (auto event : start_run) {
      cudaEventDestroy(event);
    }
    for (auto event : stop_run) {
      cudaEventDestroy(event);
    }

    for (auto g : graph_main) {
      if (g)
        cudaGraphExecDestroy(g);
    }
    for (auto event : accumulate_done) {
      if (event)
        cudaEventDestroy(event);
    }
    if (visibilities_reset_done)
      cudaEventDestroy(visibilities_reset_done);
  };

  void dump_visibilities(const uint64_t end_seq_num = 0) override {

    INFO_LOG("Dumping correlations to host...");
    int current_num_integrated_units_processed =
        num_correlation_units_integrated;
    INFO_LOG("Current num integrated units processed is {}",
             current_num_integrated_units_processed);
    // GPU-side wait (not a host-blocking sync): make streams[0]'s upcoming
    // memcpy of d_visibilities_accumulator wait for every buffer's most
    // recent accumulate_visibilities to land first. accumulate_done is a
    // no-op wait until it has been recorded at least once.
    for (auto &event : accumulate_done) {
      cudaStreamWaitEvent(streams[0], event, 0);
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
                      streams[0]);
      auto *output_ctx = new OutputTransferCompleteContext{
          .output = this->output_, .block_index = block_num};

      cudaLaunchHostFunc(streams[0],
                         output_visibilities_transfer_complete_host_func,
                         output_ctx);
    }
    cudaMemsetAsync(d_visibilities_accumulator, 0, sizeof(TrimmedVisibilities),
                    streams[0]);
    num_correlation_units_integrated.store(0);
    // Record the reset so every buffer's next accumulate_visibilities (via
    // the wait above, in execute_pipeline) is ordered after it.
    cudaEventRecord(visibilities_reset_done, streams[0]);
  };
};
