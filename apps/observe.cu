#include "spatial/common.hpp"

#ifndef NUMBER_BEAMS
#define NUMBER_BEAMS 1
#endif

void writeVectorToCSV(const std::vector<float> &times,
                      const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << "\n";
    return;
  }

  // Write CSV header
  file << "index,time\n";

  // Write data
  for (size_t i = 0; i < times.size(); ++i) {
    file << i << "," << times[i] << "\n";
  }

  file.close();
  std::cout << "Data successfully written to " << filename << "\n";
}

int main(int argc, char *argv[]) {
  std::cout << "Starting....\n";
  argparse::ArgumentParser program("pipeline");
  CommonArgs args = parse_common_args(program, argc, argv);

  std::signal(SIGINT, signal_handler);

  auto logger = setup_logger(args.debug_logging);
  constexpr int num_buffers = NR_OBSERVING_BUFFERS;
  constexpr int nr_fpga_sources = NR_OBSERVING_FPGA_SOURCES;
  constexpr size_t num_packet_buffers = 24;
  constexpr int num_lambda_channels = NR_OBSERVING_CHANNELS;
  constexpr int nr_lambda_polarizations = 2;
  constexpr int nr_lambda_receivers_per_packet =
      NR_OBSERVING_RECEIVERS_PER_PACKET;
  constexpr int nr_lambda_receivers =
      nr_lambda_receivers_per_packet * nr_fpga_sources;
  constexpr int nr_lambda_padded_receivers = NR_OBSERVING_PADDED_RECEIVERS;
  constexpr int nr_lambda_padded_receivers_per_block =
      NR_OBSERVING_PADDED_RECEIVERS_PER_BLOCK;
  constexpr int nr_lambda_beams = NUMBER_BEAMS;
  constexpr int nr_lambda_time_steps_per_packet = 64;
  constexpr int nr_lambda_packets_for_correlation =
      NR_OBSERVING_PACKETS_FOR_CORRELATION; // 256
  constexpr int nr_correlation_blocks_to_integrate =
      NR_OBSERVING_CORRELATION_BLOCKS_TO_INTEGRATE; // 56
  // Power of two so the compile-time modulo in ProcessorState reduces to a
  // mask.  
  constexpr size_t PACKET_RING_BUFFER_SIZE = 1 << 19;
  using Config =
      LambdaConfig<num_lambda_channels, nr_fpga_sources,
                   nr_lambda_time_steps_per_packet, nr_lambda_receivers,
                   nr_lambda_polarizations, nr_lambda_receivers_per_packet,
                   nr_lambda_packets_for_correlation, nr_lambda_beams,
                   nr_lambda_padded_receivers,
                   nr_lambda_padded_receivers_per_block,
                   nr_correlation_blocks_to_integrate, true, 256>;

  if (args.fpga_id_vec.size() != nr_fpga_sources ||
      args.fpga_ids.size() != nr_fpga_sources) {
    throw std::runtime_error("The number of network interfaces does not match "
                             "number of FPGA sources.");
  }
  auto fpga_delays = build_fpga_delay_array<nr_fpga_sources>(args, true);

  auto gains = get_gains_structure<Config>(args);
  ProcessorState<Config, num_packet_buffers, PACKET_RING_BUFFER_SIZE> state(
      nr_lambda_packets_for_correlation, nr_lambda_time_steps_per_packet,
      args.min_freq_channel, fpga_delays, args.fpga_ids);

  // const char *beam_filename = "hdf5_trial.hdf5";
  // std::string beam_filename = "/tmp/hdf5_trial.hdf5";
  // std::string vis_filename = "hdf5_trial_vis.hdf5";
  // hid_t beam_file =
  //    H5Fcreate(beam_filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  //  HighFive::File beam_file(beam_filename, HighFive::File::Truncate);

  if (!program.is_used("-v")) {
    args.output_filename =
        make_default_filename("visibilities", args.min_freq_channel,
                              num_lambda_channels, args.fpga_id_vec);
  }
  HighFive::File vis_file(args.output_filename, HighFive::File::Truncate);
  // auto beam_writer = std::make_unique<
  //     HDF5RawBeamWriter<Config::BeamOutputType, Config::ArrivalsOutputType>>(
  //    beam_file);
  //  auto beam_writer = std::make_unique<BatchedHDF5BeamWriter<
  //    Config::BeamOutputType, Config::ArrivalsOutputType>>(beam_file, 100);
  auto beam_writer = nullptr;
  // auto beam_writer = std::make_unique<
  //     BinaryRawBeamWriter<Config::BeamOutputType,
  //     Config::ArrivalsOutputType>>(
  //    beam_filename);
  //  auto vis_writer = std::make_unique<
  //      UVFITSVisibilitiesWriter<Config::VisibilitiesOutputType>>(
  //      vis_filename, Config::NR_CHANNELS, Config::NR_POLARIZATIONS,
  //      Config::NR_PADDED_RECEIVERS, 1.0, 1.0, 1.0, 1.0);

  auto vis_writer =
      std::make_unique<HDF5VisibilitiesWriter<Config::VisibilitiesOutputType>>(
          vis_file, args.min_freq_channel,
          args.min_freq_channel + num_lambda_channels - 1,
          &args.antenna_mapping);

  auto eigen_filename =
      make_default_filename("eigendata", args.min_freq_channel,
                            num_lambda_channels, args.fpga_id_vec);
  HighFive::File eigen_file(eigen_filename, HighFive::File::Truncate);
  auto eigen_writer = nullptr; // std::make_unique<HDF5EigenWriter<
  //      Config::EigenvalueOutputType,
  //      Config::EigenvectorOutputType>>(eigen_file);

  auto fft_writer = std::make_unique<RedisBeamFFTWriter<Config::FFTOutputType>>(
      num_lambda_channels, nr_lambda_beams, nr_lambda_polarizations, "",
      100, args.redis_channels_per_write);
  // auto fft_writer = nullptr;

  auto output = std::make_shared<BufferedOutput<Config>>(
      std::move(beam_writer), std::move(vis_writer), std::move(eigen_writer),
      std::move(fft_writer));

  BeamWeightsT<Config> h_weights;

  for (auto i = 0; i < num_lambda_channels; ++i) {
    for (auto j = 0; j < nr_lambda_receivers; ++j) {
      // Null inputs -- receivers mapped to a negative antenna ID (-100 in
      // AntennaMapRegistry, FPGA streams with no antenna connected) -- get
      // zero weight so their noise is never summed into a beam. This covers
      // the unsteered case; when steering is active, compute_steering_weights
      // zeroes them the same way on the first refresh.
      const auto mapping_it = args.antenna_mapping.find(j);
      const bool null_input = mapping_it != args.antenna_mapping.end() &&
                              mapping_it->second < 0;
      const __half amplitude = __float2half(null_input ? 0.0f : 1.0f);
      for (auto k = 0; k < nr_lambda_beams; ++k) {
        for (auto l = 0; l < nr_lambda_polarizations; ++l) {
          h_weights.weights[i][l][k][j] =
              std::complex<__half>(amplitude, __float2half(0.0f));
        }
      }
    }
  }

  // Fold calibration into the synthesized steering weights instead of also
  // applying it at ingest via set_antenna_gains()/d_gains -- doing both would
  // square the correction (see compute_steering_weights() in pipeline.hpp).
  const bool fold_calibration_into_steering =
      !args.beam_targets.empty() && args.apply_gains;

  BeamSteering<Config> beam_steering(
      args.beam_targets, args.antenna_positions, args.antenna_mapping,
      args.frequency_plan, args.min_freq_channel, args.array_location,
      args.steering_update_interval_seconds, num_buffers,
      fold_calibration_into_steering ? &gains : nullptr);

  const int integration_blocks =
      args.nr_integration_blocks > 0 ? args.nr_integration_blocks
                                     : nr_correlation_blocks_to_integrate;

  LambdaGPUPipeline<Config> pipeline(num_buffers, &h_weights,
                                     std::move(beam_steering),
                                     integration_blocks);

  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);
  pipeline.set_output(output);
  if (args.apply_gains) {
    if (fold_calibration_into_steering) {
      std::cout << "Folding calibration gains into synthesized steering "
                   "weights (skipping separate ingest-time application via "
                   "set_antenna_gains to avoid double-applying)"
                << std::endl;
    } else {
      std::cout << "Applying gains as -a is selected!" << std::endl;
      pipeline.set_antenna_gains((std::complex<float> *)gains.data());
    }
  } else {
    std::cout << "Not applying gains as -a is not selected" << std::endl;
  }

  std::thread processor([&state]() { state.process_packets(); });
  std::thread pipeline_feeder([&state]() { state.pipeline_feeder(); });

  output->start_writer_loop();

  auto capture = make_packet_captures(args);
  // Activate the lock-free strided producer path: each capture thread i owns
  // ring slots i, i+N, i+2N, ... and claims them without producer_mutex.
  state.nr_capture_threads = static_cast<int>(capture.size());
  INFO_LOG("Ring buffer size: {} packets\n", PACKET_RING_BUFFER_SIZE);
  INFO_LOG("Starting threads....");
  std::vector<std::thread> receiver_threads;
  for (auto i = 0; i < (int)capture.size(); ++i) {
    receiver_threads.emplace_back(
        [&capture, &state, i]() { capture[i]->get_packets(state); });
  }

  std::cout << "Setup completed. Ready to receive!" << std::endl;
  // Print statistics periodically
  int64_t packets_received = 0;
  int timeout = 0;
  const auto start_time = std::chrono::steady_clock::now();
  while (state.running) {
    sleep(5);
    // This is nice to see outside of log files.
    uint32_t total_drops = 0;
    for (const auto &c : capture) total_drops += c->get_drops();
    std::cout << "Stats: Received=" << state.packets_received
              << ", Processed=" << state.packets_processed
              << ", Missing=" << state.packets_missing
              << ", Discarded=" << state.packets_discarded
              << ", FutureQueued=" << state.packets_future_queued
              << ", StuckUnprocessed=" << state.packets_stuck_unprocessed
              << ", NICDrops=" << total_drops
              << std::endl;
    std::cout << "Pipeline Runs Queued = " << state.pipeline_runs_queued
              << std::endl;
    state.running.store((int)running, std::memory_order_release);
    // This is my attempt at a rudimentary shutdown procedure
    // when there are no more packets running through in a 20sec period.
    if (packets_received != 0) {
      if (packets_received == state.packets_received) {
        std::cout
            << "Packets received is same as state... adding to timeout.\n";
        timeout += 1;
      } else {
        std::cout << "Packets received is " << packets_received
                  << " and state.packets_received is " << state.packets_received
                  << ".\n";
        timeout = 0;
      }
      if (timeout > 4) {
        std::cout << "Timeout reached...shutting down\n";
        state.running.store(0, std::memory_order_release);
        running = false;
      }
    }
    packets_received = state.packets_received;

    if (args.packets_to_receive > 0 &&
        packets_received >= args.packets_to_receive) {
      std::cout << "Number of packets to observe reached...shutting down\n";
      state.running.store(0, std::memory_order_release);
      running = false;
    }
    if (args.run_duration_seconds > 0) {
      const auto elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start_time).count();
      if (elapsed >= args.run_duration_seconds) {
        std::cout << "Duration limit reached...shutting down\n";
        state.running.store(0, std::memory_order_release);
        running = false;
      }
    }
  }

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
  std::vector<float> run_timings;
  run_timings.reserve(pipeline.NR_BENCHMARKING_RUNS);
  for (auto i = 0; i < pipeline.NR_BENCHMARKING_RUNS; ++i) {
    float ms;
    cudaEventElapsedTime(&ms, pipeline.start_run[i], pipeline.stop_run[i]);
    if (ms != 0.0f) {
      run_timings.push_back(ms);
    };
  }
  writeVectorToCSV(run_timings, "output_timings.csv");
  FLUSH_LOG();
  spdlog::shutdown();
  std::cout << "Shutdown complete.\n";
  return 0;
}
