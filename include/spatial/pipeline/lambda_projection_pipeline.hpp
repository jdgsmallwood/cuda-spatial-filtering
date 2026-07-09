#pragma once

template <typename T, int NR_SIGNAL_EIGENVECTORS, int NR_RUNS_TO_AVERAGE>
class LambdaProjectionPipeline : public GPUPipeline {

  static_assert(NR_SIGNAL_EIGENVECTORS >= 1 &&
                    NR_SIGNAL_EIGENVECTORS <= T::NR_RECEIVERS,
                "NR_SIGNAL_EIGENVECTORS must be in [1, NR_RECEIVERS]");
  static_assert(NR_RUNS_TO_AVERAGE >= 1, "NR_RUNS_TO_AVERAGE must be >= 1");

private:
  // -------------------------------------------------------------------------
  // Compile-time constants
  // -------------------------------------------------------------------------
  static constexpr int NR_TIMES_PER_BLOCK = 128 / 16; // fp16 pipeline
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

  // cuSOLVER batch size: one (NR_RECEIVERS × NR_RECEIVERS) matrix per
  // channel × pol × pol, matching LambdaGPUPipeline exactly.
  static constexpr int CUSOLVER_BATCH_SIZE =
      T::NR_CHANNELS * T::NR_POLARIZATIONS * T::NR_POLARIZATIONS;

  // cuBLAS batch size for the cherk calls (same logical grouping).
  static constexpr int CUBLAS_BATCH_SIZE = CUSOLVER_BATCH_SIZE;

  inline static const __half alpha_16 = __float2half(1.0f);
  static constexpr float alpha_32 = 1.0f;

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
                         [T::NR_POLARIZATIONS][T::NR_POLARIZATIONS];

  // Full NR_RECEIVERS × NR_RECEIVERS matrices (one per channel × pol × pol),
  // laid out as a flat batch for cuSOLVER / cuBLAS.
  // Shape: [CUSOLVER_BATCH_SIZE][NR_RECEIVERS][NR_RECEIVERS]
  using DecompositionVisibilities =
      std::complex<float>[T::NR_CHANNELS][T::NR_POLARIZATIONS]
                         [T::NR_POLARIZATIONS][T::NR_RECEIVERS]
                         [T::NR_RECEIVERS];

  // Eigenvalues: one real vector of length NR_RECEIVERS per batch element.
  using Eigenvalues = float[T::NR_CHANNELS][T::NR_POLARIZATIONS]
                           [T::NR_POLARIZATIONS][T::NR_RECEIVERS];

  // Accumulated projection matrix P_acc = sum_k U_k U_k^H, same shape as the
  // full correlation matrix.
  using ProjectionAccumulator = DecompositionVisibilities;

  // -------------------------------------------------------------------------
  // Tensor-mode labels (single-character axis identifiers)
  //
  // a = unpadded baselines
  // b = block
  // c = channel
  // d = padded receivers
  // f = fpga
  // l = baseline (padded)
  // n = receivers per packet
  // o = packets for correlation
  // p = polarization
  // q = second polarization
  // r = receiver
  // s = time consolidated (block × time)
  // t = times per block
  // u = time steps per packet
  // z = complex (real, imaginary)
  // -------------------------------------------------------------------------
  inline static const std::vector<int> modePacket{'c', 'o', 'f', 'u',
                                                  'n', 'p', 'z'};
  inline static const std::vector<int> modePacketPadding{'f', 'n', 'c', 'o',
                                                         'u', 'p', 'z'};
  inline static const std::vector<int> modePacketPadded{'d', 'c', 'b',
                                                        't', 'p', 'z'};
  inline static const std::vector<int> modeCorrelatorInput{'c', 'b', 'd',
                                                           'p', 't', 'z'};
  inline static const std::vector<int> modeVisCorr{'c', 'l', 'p', 'q', 'z'};
  inline static const std::vector<int> modeVisCorrBaseline{'l', 'c', 'p', 'q',
                                                           'z'};
  inline static const std::vector<int> modeVisCorrBaselineTrimmed{'a', 'c', 'p',
                                                                  'q', 'z'};
  inline static const std::vector<int> modeVisCorrTrimmed{'c', 'a', 'p', 'q',
                                                          'z'};
  inline static const std::vector<int> modeVisDecomp{'c', 'p', 'q', 'a', 'z'};

  inline static const std::unordered_map<int, int64_t> extent = {
      {'a', NR_UNPADDED_BASELINES},
      {'b', NR_BLOCKS_FOR_CORRELATION},
      {'c', T::NR_CHANNELS},
      {'d', T::NR_PADDED_RECEIVERS},
      {'f', T::NR_FPGA_SOURCES},
      {'l', NR_BASELINES},
      {'n', T::NR_RECEIVERS_PER_PACKET},
      {'o', T::NR_PACKETS_FOR_CORRELATION},
      {'p', T::NR_POLARIZATIONS},
      {'q', T::NR_POLARIZATIONS},
      {'r', T::NR_RECEIVERS},
      {'s', NR_BLOCKS_FOR_CORRELATION *NR_TIMES_PER_BLOCK},
      {'t', NR_TIMES_PER_BLOCK},
      {'u', T::NR_TIME_STEPS_PER_PACKET},
      {'z', 2},
  };

  // -------------------------------------------------------------------------
  // Per-buffer resources (RAII, move-only)
  // -------------------------------------------------------------------------
  struct PipelineResources {
    cudaStream_t stream = nullptr;
    cudaStream_t host_stream = nullptr;

    // Raw samples
    DevicePtr<typename T::InputPacketSamplesType> samples_entry;
    DevicePtr<typename T::PacketScalesType> scales;
    DevicePtr<typename T::HalfPacketSamplesType> samples_half;
    DevicePtr<typename T::HalfPacketSamplesType> samples_padding;
    DevicePtr<typename T::PaddedPacketSamplesType> samples_padded;

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

    // cuBLAS handle for cherk (UU^H accumulation)
    cublasHandle_t cublas_handle = nullptr;

    // Per-block projection matrix P = U U^H, written by cherk before being
    // added to the shared accumulator.
    DevicePtr<DecompositionVisibilities> projection_block;

    // -----------------------------------------------------------------------
    PipelineResources() = default;

    explicit PipelineResources(
        const DecompositionVisibilities *decomp_for_workspace_query)
        : samples_entry(make_device_ptr<typename T::InputPacketSamplesType>()),
          scales(make_device_ptr<typename T::PacketScalesType>()),
          samples_half(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_padding(make_device_ptr<typename T::HalfPacketSamplesType>()),
          samples_padded(
              make_device_ptr<typename T::PaddedPacketSamplesType>()),
          correlator_input(make_device_ptr<CorrelatorInput>()),
          correlator_output(make_device_ptr<CorrelatorOutput>()),
          visibilities_baseline(make_device_ptr<Visibilities>()),
          visibilities_trimmed_baseline(make_device_ptr<TrimmedVisibilities>()),
          visibilities_trimmed(make_device_ptr<TrimmedVisibilities>()),
          decomp_visibilities(make_device_ptr<DecompositionVisibilities>()),
          eigenvalues(make_device_ptr<Eigenvalues>()),
          cusolver_info(
              make_device_ptr<int>(CUSOLVER_BATCH_SIZE * sizeof(int))),
          projection_block(make_device_ptr<DecompositionVisibilities>()) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUDA_CHECK(
          cudaStreamCreateWithFlags(&host_stream, cudaStreamNonBlocking));

      // cuSOLVER
      CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
      CUSOLVER_CHECK(cusolverDnCreateParams(&cusolver_params));
      CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle, stream));

      CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
          cusolver_handle, cusolver_params, CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_UPPER, T::NR_RECEIVERS, CUDA_C_32F,
          decomp_for_workspace_query, T::NR_RECEIVERS, CUDA_R_32F, nullptr,
          CUDA_C_32F, &cusolver_work_device_size, &cusolver_work_host_size,
          CUSOLVER_BATCH_SIZE));

      cusolver_work_device = make_device_ptr<void>(cusolver_work_device_size);
      cusolver_work_host = std::malloc(cusolver_work_host_size);

      // cuBLAS
      CUBLAS_CHECK(cublasCreate(&cublas_handle));
      CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    }

    ~PipelineResources() {
      if (stream)
        cudaStreamDestroy(stream);
      if (host_stream)
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

    // Move-only
    PipelineResources(PipelineResources &&o) noexcept
        : stream(o.stream), host_stream(o.host_stream),
          samples_entry(std::move(o.samples_entry)),
          scales(std::move(o.scales)), samples_half(std::move(o.samples_half)),
          samples_padding(std::move(o.samples_padding)),
          samples_padded(std::move(o.samples_padded)),
          correlator_input(std::move(o.correlator_input)),
          correlator_output(std::move(o.correlator_output)),
          visibilities_baseline(std::move(o.visibilities_baseline)),
          visibilities_trimmed_baseline(
              std::move(o.visibilities_trimmed_baseline)),
          visibilities_trimmed(std::move(o.visibilities_trimmed)),
          decomp_visibilities(std::move(o.decomp_visibilities)),
          eigenvalues(std::move(o.eigenvalues)),
          cusolver_handle(o.cusolver_handle),
          cusolver_params(o.cusolver_params),
          cusolver_info(std::move(o.cusolver_info)),
          cusolver_work_device(std::move(o.cusolver_work_device)),
          cusolver_work_host(o.cusolver_work_host),
          cusolver_work_device_size(o.cusolver_work_device_size),
          cusolver_work_host_size(o.cusolver_work_host_size),
          cublas_handle(o.cublas_handle),
          projection_block(std::move(o.projection_block)) {
      o.stream = nullptr;
      o.host_stream = nullptr;
      o.cusolver_handle = nullptr;
      o.cusolver_params = nullptr;
      o.cusolver_work_host = nullptr;
      o.cublas_handle = nullptr;
    }

    PipelineResources &operator=(PipelineResources &&o) noexcept {
      if (this != &o) {
        this->~PipelineResources();
        new (this) PipelineResources(std::move(o));
      }
      return *this;
    }

    PipelineResources(const PipelineResources &) = delete;
    PipelineResources &operator=(const PipelineResources &) = delete;
  };

  // -------------------------------------------------------------------------
  // Members
  // -------------------------------------------------------------------------
  int num_buffers;
  int current_buffer = 0;
  std::vector<PipelineResources> buffers;

  tcc::Correlator correlator;
  CutensorSetup tensor_16;
  CutensorSetup tensor_32;

  // Shared projection accumulator (lives outside per-buffer resources, exactly
  // like d_visibilities_accumulator in LambdaGPUPipeline).
  DecompositionVisibilities *d_projection_accumulator = nullptr;

  // Final averaged projection matrix. cuSOLVER overwrites this in-place with
  // eigenvectors, which are then transferred to the Output landing pointer.
  DecompositionVisibilities *d_projection_averaged = nullptr;

  // Scratch eigenvalue buffer for the final cuSOLVER call — allocated once
  // since dump_projection is serialised by cudaDeviceSynchronize.
  // These eigenvalues are NOT exported; only the eigenvectors are.
  Eigenvalues *d_projection_eigenvalues_scratch = nullptr;

  std::atomic<int> num_runs_integrated{0};

  typename T::AntennaGains *d_gains;
  cusolverEigMode_t cusolver_jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t cusolver_uplo = CUBLAS_FILL_MODE_UPPER;

public:
  // -------------------------------------------------------------------------
  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {

    if (!dummy_run && state_ == nullptr) {
      throw std::logic_error("State has not been set on GPUPipeline object!");
    }

    const uint64_t start_seq_num = packet_data->start_seq_id;
    const uint64_t end_seq_num = packet_data->end_seq_id;
    INFO_LOG("LambdaProjectionPipeline run: start_seq={} end_seq={}",
             start_seq_num, end_seq_num);

    auto &b = buffers[current_buffer];

    LambdaPipelineIngest<T>::ingest_and_scale(
        this->state_, packet_data, b.stream, b.host_stream,
        b.samples_entry.get(), b.scales.get(), d_gains, b.samples_half.get(),
        dummy_run);
    tensor_16.runPermutation(
        "packetToPadding", alpha_16,
        reinterpret_cast<__half *>(b.samples_half.get()),
        reinterpret_cast<__half *>(b.samples_padding.get()), b.stream);

    CUDA_CHECK(cudaMemcpyAsync(b.samples_padded.get(), b.samples_padding.get(),
                               sizeof(typename T::HalfPacketSamplesType),
                               cudaMemcpyDefault, b.stream));
    CUDA_CHECK(
        cudaMemsetAsync(reinterpret_cast<char *>(b.samples_padded.get()) +
                            sizeof(typename T::HalfPacketSamplesType),
                        0,
                        sizeof(typename T::PaddedPacketSamplesType) -
                            sizeof(typename T::HalfPacketSamplesType),
                        b.stream));

    tensor_16.runPermutation(
        "paddedToCorrInput", alpha_16,
        reinterpret_cast<__half *>(b.samples_padded.get()),
        reinterpret_cast<__half *>(b.correlator_input.get()), b.stream);
    correlator.launchAsync(
        static_cast<CUstream>(b.stream),
        reinterpret_cast<CUdeviceptr>(b.correlator_output.get()),
        reinterpret_cast<CUdeviceptr>(b.correlator_input.get()));

    tensor_32.runPermutation(
        "visCorrToBaseline", alpha_32,
        reinterpret_cast<float *>(b.correlator_output.get()),
        reinterpret_cast<float *>(b.visibilities_baseline.get()), b.stream);

    CUDA_CHECK(cudaMemcpyAsync(
        b.visibilities_trimmed_baseline.get(), b.visibilities_baseline.get(),
        sizeof(TrimmedVisibilities), cudaMemcpyDefault, b.stream));

    tensor_32.runPermutation(
        "visBaselineTrimmedToTrimmed", alpha_32,
        reinterpret_cast<float *>(b.visibilities_trimmed_baseline.get()),
        reinterpret_cast<float *>(b.visibilities_trimmed.get()), b.stream);

    // ------------------------------------------------------------------
    // 9. Expand triangular baselines → full NR_RECEIVERS × NR_RECEIVERS
    //    Hermitian matrices (one per channel × pol × pol).
    // ------------------------------------------------------------------
    tensor_32.runPermutation(
        "visCorrToDecomp", alpha_32,
        reinterpret_cast<float *>(b.visibilities_trimmed.get()),
        reinterpret_cast<float *>(b.decomp_visibilities.get()), b.stream);

    unpack_triangular_baseline_batch_launch<cuComplex>(
        reinterpret_cast<cuComplex *>(b.decomp_visibilities.get()),
        reinterpret_cast<cuComplex *>(b.decomp_visibilities.get()),
        T::NR_RECEIVERS, CUSOLVER_BATCH_SIZE, T::NR_CHANNELS, b.stream);

    // ------------------------------------------------------------------
    // 10. Eigen-decompose R per batch element.
    //     cuSOLVER overwrites decomp_visibilities with eigenvectors
    //     (columns, ascending eigenvalue order).
    // ------------------------------------------------------------------
    CUSOLVER_CHECK(cusolverDnXsyevBatched(
        b.cusolver_handle, b.cusolver_params, cusolver_jobz, cusolver_uplo,
        T::NR_RECEIVERS, CUDA_C_32F,
        reinterpret_cast<void *>(b.decomp_visibilities.get()), T::NR_RECEIVERS,
        CUDA_R_32F, reinterpret_cast<void *>(b.eigenvalues.get()), CUDA_C_32F,
        b.cusolver_work_device.get(), b.cusolver_work_device_size,
        b.cusolver_work_host, b.cusolver_work_host_size, b.cusolver_info.get(),
        CUSOLVER_BATCH_SIZE));

    {
      constexpr int N = T::NR_RECEIVERS;
      constexpr int K = NR_SIGNAL_EIGENVECTORS;
      constexpr int col_offset = N - K; // first signal-subspace column
      const float herk_alpha = 1.0f;
      const float herk_beta = 0.0f; // overwrite projection_block

      auto *V_base = reinterpret_cast<cuComplex *>(b.decomp_visibilities.get());
      auto *P_base = reinterpret_cast<cuComplex *>(b.projection_block.get());

      for (int batch = 0; batch < CUBLAS_BATCH_SIZE; ++batch) {
        // Pointer to the start of eigenvector matrix for this batch element.
        cuComplex *V_batch = V_base + batch * N * N;
        // Pointer to signal-subspace U = last K columns of V_batch.
        cuComplex *U = V_batch + col_offset * N;
        // Pointer to output P for this batch element.
        cuComplex *P_batch = P_base + batch * N * N;

        CUBLAS_CHECK(
            cublasCherk(b.cublas_handle,
                        CUBLAS_FILL_MODE_UPPER, // fill upper triangle of P
                        CUBLAS_OP_N,            // no transpose on U
                        N, K, &herk_alpha, U, N, &herk_beta, P_batch, N));
      }
    }

    // ------------------------------------------------------------------
    // 12. Add P_block to the shared accumulator.
    //     accumulate_visibilities adds src into dst element-wise on the
    //     stream (matching usage in LambdaGPUPipeline).
    // ------------------------------------------------------------------
    accumulate_visibilities(
        reinterpret_cast<float *>(b.projection_block.get()),
        reinterpret_cast<float *>(d_projection_accumulator),
        // Factor of 2 for complex (real + imag); full square matrix.
        2 * CUSOLVER_BATCH_SIZE * T::NR_RECEIVERS * T::NR_RECEIVERS, b.stream);

    // ------------------------------------------------------------------
    // 13. After NR_RUNS_TO_AVERAGE blocks, average, decompose and export.
    // ------------------------------------------------------------------
    if (!dummy_run) {
      num_runs_integrated.fetch_add(1);
      if (num_runs_integrated.load() >= NR_RUNS_TO_AVERAGE) {
        dump_projection(start_seq_num, end_seq_num);
      }
    }

    // ------------------------------------------------------------------
    // 14. Rotate buffer index.
    // ------------------------------------------------------------------
    if (!dummy_run) {
      current_buffer = (current_buffer + 1) % num_buffers;
    }
  }

  // -------------------------------------------------------------------------
  // dump_visibilities — public override required by GPUPipeline.
  //
  // Called externally (e.g. on shutdown) to flush whatever has accumulated.
  // We delegate to dump_projection with end_seq_num = 0 (unknown) matching
  // the convention in LambdaGPUPipeline.
  // -------------------------------------------------------------------------
  void dump_visibilities(const uint64_t end_seq_num = 0) override {
    if (num_runs_integrated.load() > 0) {
      dump_projection(/* start */ 0, end_seq_num);
    }
  }

  // -------------------------------------------------------------------------
  // Constructor
  // -------------------------------------------------------------------------
  LambdaProjectionPipeline(const int num_buffers_in)
      : num_buffers(num_buffers_in),

        correlator(cu::Device(0), tcc::Format::fp16, T::NR_PADDED_RECEIVERS,
                   T::NR_CHANNELS,
                   NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK,
                   T::NR_POLARIZATIONS, std::nullopt,
                   T::NR_PADDED_RECEIVERS_PER_BLOCK,
                   TCC_THREAD_BLOCKS_PER_SM),

        tensor_16(extent, CUTENSOR_R_16F, 128),
        tensor_32(extent, CUTENSOR_R_32F, 128) {
    std::cout << "LambdaProjectionPipeline instantiated:"
              << " NR_CHANNELS=" << T::NR_CHANNELS
              << " NR_RECEIVERS=" << T::NR_PADDED_RECEIVERS
              << " NR_POLARIZATIONS=" << T::NR_POLARIZATIONS
              << " NR_SAMPLES_PER_CH="
              << NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK
              << " NR_SIGNAL_EIGENVECTORS=" << NR_SIGNAL_EIGENVECTORS
              << " NR_RUNS_TO_AVERAGE=" << NR_RUNS_TO_AVERAGE << std::endl;

    // Allocate shared accumulator, averaged-projection, and scratch eigenvalue
    // buffers.  The eigenvalue scratch is needed by the final cuSOLVER call
    // but is NOT exported — only eigenvectors are transferred to the Output.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_projection_accumulator),
                          sizeof(DecompositionVisibilities)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_projection_averaged),
                          sizeof(DecompositionVisibilities)));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_projection_eigenvalues_scratch),
                   sizeof(Eigenvalues)));
    CUDA_CHECK(cudaMemset(d_projection_accumulator, 0,
                          sizeof(DecompositionVisibilities)));
    CUDA_CHECK(cudaMemset(d_projection_averaged, 0,
                          sizeof(DecompositionVisibilities)));

    CUDA_CHECK(cudaMalloc((void **)&d_gains, sizeof(typename T::AntennaGains)));
    auto default_gains = get_default_gains<T::NR_CHANNELS, T::NR_RECEIVERS,
                                           T::NR_POLARIZATIONS>();
    CUDA_CHECK(cudaMemcpy(d_gains, default_gains.data(),
                          sizeof(typename T::AntennaGains), cudaMemcpyDefault));
    // We need a valid device pointer for the cuSOLVER workspace query inside
    // PipelineResources.  Use the (already allocated) accumulator pointer;
    // the query is read-only with respect to the matrix pointer.
    buffers.reserve(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(d_projection_accumulator);
    }

    cudaDeviceSynchronize();

    // Tensor descriptors
    tensor_16.addTensor(modePacket, "packet");
    tensor_16.addTensor(modePacketPadding, "packet_padding");
    tensor_16.addTensor(modePacketPadded, "packet_padded");
    tensor_16.addTensor(modeCorrelatorInput, "corr_input");

    tensor_32.addTensor(modeVisCorr, "visCorr");
    tensor_32.addTensor(modeVisCorrBaseline, "visBaseline");
    tensor_32.addTensor(modeVisCorrBaselineTrimmed, "visBaselineTrimmed");
    tensor_32.addTensor(modeVisCorrTrimmed, "visCorrTrimmed");
    tensor_32.addTensor(modeVisDecomp, "visDecomp");

    // Permutation descriptors
    tensor_16.addPermutation("packet", "packet_padding",
                             CUTENSOR_COMPUTE_DESC_16F, "packetToPadding");
    tensor_16.addPermutation("packet_padded", "corr_input",
                             CUTENSOR_COMPUTE_DESC_16F, "paddedToCorrInput");

    tensor_32.addPermutation("visCorr", "visBaseline",
                             CUTENSOR_COMPUTE_DESC_32F, "visCorrToBaseline");
    tensor_32.addPermutation("visBaselineTrimmed", "visCorrTrimmed",
                             CUTENSOR_COMPUTE_DESC_32F,
                             "visBaselineTrimmedToTrimmed");
    tensor_32.addPermutation("visCorrTrimmed", "visDecomp",
                             CUTENSOR_COMPUTE_DESC_32F, "visCorrToDecomp");

    // Warm-up to JIT all template kernels before first real data arrives.
    typename T::PacketFinalDataType warmup_packet;
    std::memset(warmup_packet.samples, 0,
                warmup_packet.get_samples_elements_size());
    std::memset(warmup_packet.scales, 0,
                warmup_packet.get_scales_element_size());
    std::memset(warmup_packet.arrivals, 0, warmup_packet.get_arrivals_size());
    execute_pipeline(&warmup_packet, /*dummy_run=*/true);
    cudaDeviceSynchronize();

    // Reset accumulator after warm-up (dummy run may have polluted it with
    // zeros, but be explicit for correctness).
    CUDA_CHECK(cudaMemset(d_projection_accumulator, 0,
                          sizeof(DecompositionVisibilities)));
    num_runs_integrated.store(0);
  }

  // -------------------------------------------------------------------------
  // Destructor
  // -------------------------------------------------------------------------
  ~LambdaProjectionPipeline() {
    cudaFree(d_projection_accumulator);
    cudaFree(d_projection_averaged);
    cudaFree(d_projection_eigenvalues_scratch);
    // buffers destructs automatically, cleaning up per-buffer GPU memory,
    // cuSOLVER handles, and cuBLAS handles.
  }

private:
  void dump_projection(const uint64_t start_seq_num,
                       const uint64_t end_seq_num) {
    INFO_LOG("LambdaProjectionPipeline: dumping averaged projection "
             "(num_runs={})",
             num_runs_integrated.load());

    const int runs = num_runs_integrated.load();

    // Synchronise device so all pending GPU work is complete.
    cudaDeviceSynchronize();

    // Copy accumulator → averaged buffer, then divide by run count.
    // We reuse stream 0 (buffers[0].stream) for these serialised steps.
    auto &b0 = buffers[0];

    CUDA_CHECK(cudaMemcpyAsync(d_projection_averaged, d_projection_accumulator,
                               sizeof(DecompositionVisibilities),
                               cudaMemcpyDefault, b0.stream));

    // Divide each element by 'runs' (scale factor = 1/runs).
    // scale_visibilities is assumed to exist alongside accumulate_visibilities.
    scale_visibilities(reinterpret_cast<float *>(d_projection_averaged),
                       2 * CUSOLVER_BATCH_SIZE * T::NR_RECEIVERS *
                           T::NR_RECEIVERS,
                       1.0f / static_cast<float>(runs), b0.stream);

    // Eigen-decompose the averaged projection matrix.
    // cuSOLVER writes eigenvectors in-place (ascending eigenvalue order).
    // Eigenvalues land in d_projection_eigenvalues_scratch — they are NOT
    // exported.
    CUSOLVER_CHECK(cusolverDnXsyevBatched(
        b0.cusolver_handle, b0.cusolver_params, cusolver_jobz, cusolver_uplo,
        T::NR_RECEIVERS, CUDA_C_32F,
        reinterpret_cast<void *>(d_projection_averaged), T::NR_RECEIVERS,
        CUDA_R_32F, reinterpret_cast<void *>(d_projection_eigenvalues_scratch),
        CUDA_C_32F, b0.cusolver_work_device.get(), b0.cusolver_work_device_size,
        b0.cusolver_work_host, b0.cusolver_work_host_size,
        b0.cusolver_info.get(), CUSOLVER_BATCH_SIZE));

    cudaDeviceSynchronize();

    // Export eigenvectors only via the Output interface.
    if (output_ != nullptr) {
      size_t block_num = output_->register_eigendecomposition_data_block(
          start_seq_num, end_seq_num);
      // size_t::max means no eigen writer attached -- the landing pointers
      // would be nullptr.
      if (block_num != std::numeric_limits<size_t>::max()) {
        // d_projection_averaged now holds the eigenvectors (cuSOLVER
        // in-place).
        void *eigval_ptr =
            output_->get_eigenvalues_data_landing_pointer(block_num);
        void *eigvec_ptr =
            output_->get_eigenvectors_data_landing_pointer(block_num);
        CUDA_CHECK(cudaMemcpyAsync(eigvec_ptr, d_projection_averaged,
                                   sizeof(DecompositionVisibilities),
                                   cudaMemcpyDefault, b0.stream));

        CUDA_CHECK(cudaMemcpyAsync(eigval_ptr, d_projection_eigenvalues_scratch,
                                   sizeof(Eigenvalues), cudaMemcpyDefault,
                                   b0.stream));

        auto *ctx = new OutputTransferCompleteContext{.output = this->output_,
                                                      .block_index = block_num};
        CUDA_CHECK(cudaLaunchHostFunc(
            b0.stream, eigen_output_transfer_complete_host_func, ctx));
      }
    }

    // Reset accumulator and run counter.
    CUDA_CHECK(cudaMemsetAsync(d_projection_accumulator, 0,
                               sizeof(DecompositionVisibilities), b0.stream));
    num_runs_integrated.store(0);

    cudaDeviceSynchronize();
    INFO_LOG("LambdaProjectionPipeline: dump complete.");
  }
};
