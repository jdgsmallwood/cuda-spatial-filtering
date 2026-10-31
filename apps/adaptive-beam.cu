#include "spatial/common.hpp"
#include "spatial/run_audit.hpp"
#include "spatial/adaptive_calibration.hpp"

#ifndef NR_OBSERVING_FINE_CHANNELS
#define NR_OBSERVING_FINE_CHANNELS 1
#endif

#ifndef NR_OBSERVING_FINE_CHANNEL_EDGE_TRIM
#define NR_OBSERVING_FINE_CHANNEL_EDGE_TRIM 0
#endif

#ifndef SPATIAL_ADAPTIVE_BUFFERS
#define SPATIAL_ADAPTIVE_BUFFERS NR_OBSERVING_BUFFERS
#endif

template <BeamformingMode Mode>
int run_beam(CommonArgs args, const std::string &calibration, bool protect_sun,
             int null_rank, bool record_visibilities, int num_buffers, int argc, char **argv) {

  std::signal(SIGINT, signal_handler);

  auto logger = setup_logger(args.debug_logging);

  constexpr int nr_fpga_sources = NR_OBSERVING_FPGA_SOURCES;
  constexpr size_t num_packet_buffers = NR_OBSERVING_PACKET_BUFFERS;
  constexpr int num_lambda_channels = NR_OBSERVING_CHANNELS;
  constexpr int nr_lambda_polarizations = 2;
  constexpr int nr_lambda_receivers_per_packet =
      NR_OBSERVING_RECEIVERS_PER_PACKET;
  constexpr int nr_lambda_receivers =
      nr_lambda_receivers_per_packet * nr_fpga_sources;
  constexpr int nr_lambda_padded_receivers = NR_OBSERVING_PADDED_RECEIVERS;
  constexpr int nr_lambda_padded_receivers_per_block =
      NR_OBSERVING_PADDED_RECEIVERS_PER_BLOCK;
  constexpr int nr_lambda_beams = 1; // NUMBER_BEAMS;
  constexpr int nr_lambda_time_steps_per_packet = 64;
  constexpr int nr_lambda_packets_for_correlation =
      NR_OBSERVING_PACKETS_FOR_CORRELATION; // 256
  constexpr int nr_correlation_blocks_to_integrate =
      NR_OBSERVING_CORRELATION_BLOCKS_TO_INTEGRATE; // 56
  constexpr int fft_downsample_factor = 64;
  constexpr int num_lambda_fine_channels = NR_OBSERVING_FINE_CHANNELS;
  constexpr int num_lambda_fine_channel_edge_trim =
      NR_OBSERVING_FINE_CHANNEL_EDGE_TRIM;
  using Config = LambdaConfig<
      num_lambda_channels, nr_fpga_sources, nr_lambda_time_steps_per_packet,
      nr_lambda_receivers, nr_lambda_polarizations,
      nr_lambda_receivers_per_packet, nr_lambda_packets_for_correlation,
      nr_lambda_beams, nr_lambda_padded_receivers,
      nr_lambda_padded_receivers_per_block, nr_correlation_blocks_to_integrate,
      true, fft_downsample_factor, num_lambda_fine_channels,
      num_lambda_fine_channel_edge_trim>;

  // 2x as there will be original & RFI mitigated beams. Only meaningful when
  // NR_FINE_CHANNELS == 1 -- the whole-band FFT is deleted once channelized
  // (see LambdaAdaptiveBeamformedSpectraPipeline::execute_pipeline).
  using FFTOutputType =
      float[Config::NR_CHANNELS][nr_lambda_polarizations][2 * nr_lambda_beams]
           [nr_lambda_time_steps_per_packet - 10];

  // Trailing axis matches the pipeline's own (private) BeamOutput type --
  // Config::NR_TIME_STEPS_PER_FINE_CHANNEL, not the raw packets*time-per-packet
  // product.
  using BeamOutputType =
      std::complex<__half>[Config::NR_CHANNELS][nr_lambda_polarizations]
                          [2 * nr_lambda_beams]
                          [Config::NR_TIME_STEPS_PER_FINE_CHANNEL];
  if (args.fpga_id_vec.size() != nr_fpga_sources ||
      args.fpga_ids.size() != nr_fpga_sources) {
    throw std::runtime_error("The number of network interfaces does not match "
                             "number of FPGA sources.");
  }

  auto fpga_delays = build_fpga_delay_array<nr_fpga_sources>(args);

  ProcessorState<Config, num_packet_buffers, DEFAULT_PACKET_RING_BUFFER_SIZE,
                 NR_OBSERVING_PACKET_WORKER_THREADS>
      state(nr_lambda_packets_for_correlation, nr_lambda_time_steps_per_packet,
            args.min_freq_channel, fpga_delays, args.fpga_ids,
            args.packet_format);

  using Eigenvalues = float[Config::NR_CHANNELS][2][nr_lambda_receivers];
  using Eigenvectors =
      std::complex<float>[Config::NR_CHANNELS][2][nr_lambda_receivers]
                         [nr_lambda_receivers];
  constexpr int decomposition_channels = Mode == BeamformingMode::Coarse
                                             ? Config::NR_FPGA_CHANNELS
                                             : Config::NR_CHANNELS;
  struct OutputConfig : Config {
    using VisibilitiesOutputType =
        float[decomposition_channels]
             [nr_lambda_receivers * (nr_lambda_receivers + 1) / 2][2][2][2];
  };
  const bool use_canonical = !args.canonical_recv_perm.empty();
  const auto &active_mapping =
      use_canonical ? args.canonical_antenna_mapping : args.antenna_mapping;
  std::cout << "Loading weights...\n";
  BeamWeightsT<Config> h_weights;

  for (auto i = 0; i < Config::NR_CHANNELS; ++i) {
    for (auto j = 0; j < nr_lambda_receivers; ++j) {
      for (auto k = 0; k < nr_lambda_beams; ++k) {
        for (auto l = 0; l < nr_lambda_polarizations; ++l) {
          h_weights.weights[i][l][k][j] = std::complex<__half>(
              __float2half(1.0f / static_cast<float>(Config::NR_RECEIVERS)),
              __float2half(0.0f / static_cast<float>(Config::NR_RECEIVERS)));
        }
      }
    }
  }

  auto calibration_data = load_adaptive_calibration<Config>(
      args, calibration == "gains", Mode == BeamformingMode::Coarse, null_rank);
  auto &calibration_gains = calibration_data.gains;
  for (size_t c = 0; c < Config::NR_CHANNELS; ++c)
    for (size_t p = 0; p < 2; ++p)
      for (size_t r = 0; r < Config::NR_RECEIVERS; ++r) {
        const auto gain =
            calibration_gains[c][p][r] / float(Config::NR_RECEIVERS);
        h_weights.weights[c][p][0][r] = {__float2half(gain.real()),
                                         __float2half(gain.imag())};
      }
  std::cout << "Excluded receptor/channel entries: "
            << calibration_data.exclusions.size() << '\n';

  BeamSteering<Config> beam_steering(
      args.beam_targets, args.antenna_positions, active_mapping,
      args.frequency_plan, args.min_freq_channel, args.array_location,
      args.steering_update_interval_seconds, num_buffers, &calibration_gains,
      args.steering_utc);

  std::cout << "Creating Output Handler\n";
  std::string beam_filename =
      args.beam_output_filename.empty()
          ? make_default_filename("beam", args.min_freq_channel,
                                  num_lambda_channels, args.fpga_id_vec)
          : args.beam_output_filename;
  HighFive::File beam_file(beam_filename, HighFive::File::Excl);
  beam_file.createAttribute<std::string>(
      "decomposition", Mode == BeamformingMode::WeightsOnly ? "weights-only"
                       : Mode == BeamformingMode::Coarse    ? "coarse"
                                                            : "fine");
  beam_file.createAttribute<std::string>("calibration_mode", calibration);
  beam_file.createAttribute<std::string>("calibration_exclusions_json",
                                         calibration_data.exclusions.dump());
  beam_file.createAttribute<std::string>(
      "protection_model",
      protect_sun ? (calibration == "gains" ? "prior-calibrated-signature"
                                            : "assumed-geometric-signature")
                  : "none");
  beam_file.createAttribute<std::string>("sun_protection",
                                         protect_sun ? "rank-one" : "off");
  beam_file.createAttribute<int>("nulling_rank", null_rank);
  beam_file.createAttribute<int>("gpu_buffers", num_buffers);
  beam_file.createAttribute<std::string>("beam_order", "unfiltered,filtered");
  beam_file.createAttribute<int>("fine_fft_channels", Config::NR_FINE_CHANNELS);
  beam_file.createAttribute<int>("first_raw_fine_bin",
                                 Config::NR_FINE_CHANNEL_EDGE_TRIM);
  beam_file.createAttribute<int>("fine_channels_per_coarse",
                                 Config::NR_EFFECTIVE_FINE_CHANNELS);
  // Match production's scaling before the PFB narrows its output to fp16.
  constexpr float fine_voltage_scale =
      Config::NR_FINE_CHANNELS > 1 ? 1.0f / 64.0f : 1.0f;
  beam_file.createAttribute<float>("fine_voltage_scale", fine_voltage_scale);
  write_hdf5_run_audit(beam_file, args, argc, argv,
                       nr_lambda_receivers_per_packet, 2);
  auto beam_writer = std::make_unique<HDF5BeamWriter<
      BeamOutputType, Config::ArrivalsOutputType, std::nullptr_t>>(beam_file,
                                                                   8);

  std::unique_ptr<HighFive::File> vis_file;
  std::unique_ptr<
      VisibilitiesWriter<typename OutputConfig::VisibilitiesOutputType>>
      vis_writer;
  if (record_visibilities) {
    const auto filename =
        args.output_filename.empty()
            ? make_default_filename("vis", args.min_freq_channel,
                                    num_lambda_channels, args.fpga_id_vec)
            : args.output_filename;
    vis_file = std::make_unique<HighFive::File>(filename, HighFive::File::Excl);
    vis_file->createAttribute<std::string>(
        "decomposition", Mode == BeamformingMode::Coarse ? "coarse" : "fine");
    vis_file->createAttribute<std::string>("covariance_stage",
                                           "before-whitening");
    vis_file->createAttribute<std::string>("calibration_exclusions_json",
                                           calibration_data.exclusions.dump());
    vis_file->createAttribute<float>(
        "voltage_scale",
        Mode == BeamformingMode::Coarse ? 1.0f : fine_voltage_scale);
    vis_file->createAttribute<int>(
        "samples_per_covariance", Mode == BeamformingMode::Coarse
                                      ? Config::NR_PACKETS_FOR_CORRELATION *
                                            Config::NR_TIME_STEPS_PER_PACKET
                                      : Config::NR_TIME_STEPS_PER_FINE_CHANNEL);
    write_hdf5_run_audit(*vis_file, args, argc, argv,
                         nr_lambda_receivers_per_packet, 2);
    vis_writer = std::make_unique<
        HDF5VisibilitiesWriter<typename OutputConfig::VisibilitiesOutputType>>(
        *vis_file, args.min_freq_channel,
        args.min_freq_channel + Config::NR_FPGA_CHANNELS - 1, &active_mapping,
        8, 0, use_canonical);
  }

  auto output = std::make_shared<
      BufferedOutput<OutputConfig, FFTOutputType, Eigenvalues, Eigenvectors,
                     BeamOutputType, std::nullptr_t>>(
      std::move(beam_writer), std::move(vis_writer), nullptr, nullptr);

  std::cout << "Initializing pipeline...\n";
  std::unordered_map<int, int> ranks;
  for (int c = 0; c < decomposition_channels; ++c)
    ranks[args.min_freq_channel + c] = null_rank;
  AdaptiveBeamPipeline<Config, Mode> pipeline(
      num_buffers, &h_weights, ranks, args.min_freq_channel,
      std::move(beam_steering), false, false, 3.0f, 0.0, fine_voltage_scale);

  pipeline.set_enabled_inputs(calibration_data.enabled);
  if (protect_sun)
    pipeline.enable_solar_protection();
  if (record_visibilities)
    pipeline.enable_visibility_recording();
  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);
  pipeline.set_output(output);
  if (use_canonical)
    pipeline.set_stream_permutation(args.canonical_recv_perm,
                                    args.canonical_pol_perm);
  pipeline.prepare_cuda_graphs();
  beam_file.createAttribute<unsigned long long>("captured_cuda_graphs", pipeline.cuda_graph_count());
  std::cout << "Initializing packet capture...\n";
  auto capture = make_packet_captures(args, 512 * 1024 * 1024);
  if (args.pcap_filename.empty())
    state.nr_capture_threads = static_cast<int>(capture.size());
#ifdef HAVE_IBVERBS
  if (args.capture_backend == "ibverbs")
    arm_ibverbs_zero_copy_captures(capture, state);
#endif
  INFO_LOG("Ring buffer size: {} packets\n", DEFAULT_PACKET_RING_BUFFER_SIZE);
  std::cout << "Starting threads...\n";
  const auto measurement_start = std::chrono::steady_clock::now();
  std::vector<std::thread> receiver_threads;
  for (auto i = 0; i < (int)capture.size(); ++i) {
    receiver_threads.emplace_back(
        [&capture, &state, i]() { capture[i]->get_packets(state); });
  }

  std::thread processor([&state]() { state.process_packets(); });
  std::thread pipeline_feeder([&state]() { state.pipeline_feeder(); });
  // Start writer thread
  output->start_writer_loop();
  std::cout << "Setup completed. Ready to receive!" << std::endl;
  monitor_app_stats(state, capture, args);

  // Cleanup
  INFO_LOG("\nShutting down...\n");
  std::cout << "Shutting down...\n";
  state.running.store(0, std::memory_order_release);
  state.shutdown();

  std::cout << "Waiting for receivers to finish...\n";
  for (auto &t : receiver_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  std::cout << "Waiting for processor to finish...\n";
  processor.join();
  std::cout << "Waiting for pipeline feeder to finish...\n";
  pipeline_feeder.join();
  std::cout << "Dumping visibilities....\n";
  cudaDeviceSynchronize();
  pipeline.dump_visibilities();
  cudaDeviceSynchronize();

  std::cout << "Synchronizing GPU...\n";
  cudaDeviceSynchronize();

  std::cout << "Stopping writers...\n";
  output->running_ = false;
  output->stop_writers();
  beam_file.flush();
  if (vis_file)
    vis_file->flush();
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    measurement_start)
          .count();
  const double throughput = pipeline.successful_payload_bytes() / elapsed / 1e9;
  const unsigned solver_failures = pipeline.eigensolver_failures();
  beam_file.createAttribute<unsigned>("eigensolver_failures", solver_failures);
  if (vis_file)
    vis_file->createAttribute<unsigned>("eigensolver_failures",
                                        solver_failures);
  beam_file.createAttribute<double>("elapsed_seconds", elapsed);
  beam_file.createAttribute<uint64_t>("successful_input_payload_bytes",
                                      pipeline.successful_payload_bytes());
  beam_file.createAttribute<double>("input_GB_per_s", throughput);
  beam_file.createAttribute<uint64_t>("missing_input_packets",
                                      pipeline.missing_input_packets());
  beam_file.createAttribute<uint64_t>("input_packets",
                                      pipeline.input_packets());
  beam_file.createAttribute<uint64_t>("pipeline_blocks",
                                      pipeline.processed_blocks());
  const json measurement = {
      {"mode", Mode == BeamformingMode::WeightsOnly ? "weights-only"
               : Mode == BeamformingMode::Coarse    ? "coarse"
                                                    : "fine"},
      {"rank", null_rank},
      {"protected", protect_sun},
      {"record_visibilities", record_visibilities},
      {"eigensolver_failures", solver_failures},
      {"gpu_buffers", num_buffers},
      {"elapsed_seconds", elapsed},
      {"successful_input_payload_bytes", pipeline.successful_payload_bytes()},
      {"input_GB_per_s", throughput},
      {"missing_input_packets", pipeline.missing_input_packets()},
      {"input_packets", pipeline.input_packets()},
      {"pipeline_blocks", pipeline.processed_blocks()},
      {"capture_drops", get_total_capture_drops(capture)},
      {"discarded_packets", state.packets_discarded.load()}};
  std::cout << "ADAPTIVE_BENCHMARK " << measurement.dump() << '\n';
  FLUSH_LOG();
  spdlog::shutdown();
  std::cout << "Shutdown complete.\n";
  if (solver_failures)
    std::cerr << "Invalid run: eigensolver failed for " << solver_failures
              << " matrices\n";
  return solver_failures ? 1 : 0;
}

int main(int argc, char **argv) {
  try {
    argparse::ArgumentParser program("adaptive_beam");
    std::string mode = "weights-only", calibration = "ask", protection = "off";
    int rank = 1;
    int gpu_buffers = SPATIAL_ADAPTIVE_BUFFERS;
    program.add_argument("--gpu-buffers")
        .default_value(gpu_buffers)
        .scan<'i', int>()
        .store_into(gpu_buffers);
    bool record_visibilities = false;
    program.add_argument("--record-visibilities")
        .default_value(false)
        .implicit_value(true)
        .store_into(record_visibilities);
    program.add_argument("--decomposition")
        .choices("weights-only", "coarse", "fine")
        .default_value(mode)
        .store_into(mode);
    program.add_argument("--calibration")
        .choices("ask", "unity", "gains")
        .default_value(calibration)
        .store_into(calibration);
    program.add_argument("--sun-protection")
        .choices("off", "rank-one")
        .default_value(protection)
        .store_into(protection);
    program.add_argument("--null-rank")
        .default_value(rank)
        .scan<'i', int>()
        .store_into(rank);
    auto args = parse_common_args(program, argc, argv, false);
    if (gpu_buffers < 1 || gpu_buffers > NR_OBSERVING_PACKET_BUFFERS)
      throw std::invalid_argument("GPU buffer count must be positive and no larger than the input buffer count");
    if (!program.is_used("--steering-update-interval-seconds"))
      args.steering_update_interval_seconds = 1.0;
    if (!(args.steering_update_interval_seconds > 0) ||
        !std::isfinite(args.steering_update_interval_seconds))
      throw std::invalid_argument(
          "Steering update interval must be positive and finite");
    if (calibration == "ask") {
      if (std::ifstream(args.gains_filename).good()) {
        calibration = "gains";
      } else {
        if (!isatty(STDIN_FILENO))
          throw std::invalid_argument(
              "Choose --calibration gains or --calibration unity explicitly "
              "for unattended runs");
        std::cerr << "No gains file. Choose 1: unity gains with geometric Sun "
                     "protection, "
                     "2: unity gains without protection, 3: abort: "
                  << std::flush;
        std::string choice;
        std::getline(std::cin, choice);
        if (choice != "1" && choice != "2")
          throw std::invalid_argument("Observation cancelled");
        calibration = "unity";
        protection =
            choice == "1" && mode != "weights-only" ? "rank-one" : "off";
      }
    }
    if (calibration == "gains") {
      std::ifstream gains(args.gains_filename);
      if (!gains)
        throw std::invalid_argument(
            "Cannot open gains file; supply valid gains or explicitly select "
            "--calibration unity");
      args.gains = json::parse(gains);
    }
    if (rank < 0 ||
        rank >= NR_OBSERVING_FPGA_SOURCES * NR_OBSERVING_RECEIVERS_PER_PACKET)
      throw std::invalid_argument("nulling rank must be between 0 and N-1");
    if (mode == "weights-only")
      rank = 0;
    const bool protect = protection == "rank-one";
    if (record_visibilities && mode == "weights-only")
      throw std::invalid_argument("weights-only has no covariance to record");
    if (protect && mode == "weights-only")
      throw std::invalid_argument("weights-only does not apply a spatial "
                                  "filter; select --sun-protection off");
    if (args.capture_backend == "ibverbs-gpudirect")
      throw std::invalid_argument("adaptive_beam currently supports CPU-ring "
                                  "capture, not GPUDirect capture");
    if (protect || !args.beam_targets.empty() ||
        args.config.contains("array_location")) {
      if (!args.config.contains("array_location") ||
          args.antenna_positions.empty() ||
          !(args.frequency_plan.channel_bandwidth_hz > 0))
        throw std::invalid_argument(
            "Sun tracking requires site, antenna positions and frequency plan");
      if (!std::isfinite(args.array_location.latitude_deg) ||
          std::abs(args.array_location.latitude_deg) > 90 ||
          !std::isfinite(args.array_location.longitude_deg) ||
          std::abs(args.array_location.longitude_deg) > 180 ||
          !std::isfinite(args.array_location.height_m))
        throw std::invalid_argument("Invalid geodetic array location");
      const auto &mapping = args.canonical_recv_perm.empty()
                                ? args.antenna_mapping
                                : args.canonical_antenna_mapping;
      for (int r = 0;
           r < NR_OBSERVING_FPGA_SOURCES * NR_OBSERVING_RECEIVERS_PER_PACKET;
           ++r) {
        const auto antenna = mapping.at(r);
        if (antenna >= 0 && !args.antenna_positions.count(antenna))
          throw std::invalid_argument(
              "missing antenna position for Sun tracking");
      }
      if (!args.pcap_filename.empty() && args.steering_utc.empty())
        throw std::invalid_argument(
            "Sun-tracked replay requires --steering-utc for the first input "
            "sample");
      if (args.beam_targets.empty())
        args.beam_targets = {BeamTarget{"sun", 0, 0}};
      if (protect &&
          (args.beam_targets.size() != 1 || args.beam_targets[0].mode != "sun"))
        throw std::invalid_argument(
            "Sun protection requires exactly one Sun target");
    }
    if (mode == "weights-only")
      return run_beam<BeamformingMode::WeightsOnly>(
          args, calibration, protect, rank, record_visibilities, gpu_buffers, argc, argv);
    if (mode == "coarse")
      return run_beam<BeamformingMode::Coarse>(args, calibration, protect, rank,
                                               record_visibilities, gpu_buffers, argc, argv);
    return run_beam<BeamformingMode::Fine>(args, calibration, protect, rank,
                                           record_visibilities, gpu_buffers, argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "adaptive_beam: " << error.what() << std::endl;
    return 1;
  }
}
