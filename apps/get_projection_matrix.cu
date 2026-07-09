#include "spatial/common.hpp"

int main(int argc, char *argv[]) {
  std::cout << "Starting....\n";
  argparse::ArgumentParser program("pipeline");

  CommonArgs args = parse_common_args(program, argc, argv);

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
  constexpr int nr_lambda_beams = 1;
  constexpr int nr_lambda_time_steps_per_packet = 64;
  constexpr int nr_lambda_packets_for_correlation =
      NR_OBSERVING_PACKETS_FOR_CORRELATION; // 256
  constexpr int nr_correlation_blocks_to_integrate =
      NR_OBSERVING_CORRELATION_BLOCKS_TO_INTEGRATE; // 56
  using Config =
      LambdaConfig<num_lambda_channels, nr_fpga_sources,
                   nr_lambda_time_steps_per_packet, nr_lambda_receivers,
                   nr_lambda_polarizations, nr_lambda_receivers_per_packet,
                   nr_lambda_packets_for_correlation, nr_lambda_beams,
                   nr_lambda_padded_receivers,
                   nr_lambda_padded_receivers_per_block,
                   nr_correlation_blocks_to_integrate, true>;

  if (args.fpga_id_vec.size() != nr_fpga_sources ||
      args.fpga_ids.size() != nr_fpga_sources) {
    throw std::runtime_error("The number of network interfaces does not match "
                             "number of FPGA sources.");
  }

  std::array<int64_t, nr_fpga_sources> fpga_delays;
  for (auto i = 0; i < nr_fpga_sources; ++i) {
    fpga_delays[i] = 0;
  }
  ProcessorState<Config, num_packet_buffers, DEFAULT_PACKET_RING_BUFFER_SIZE>
      state(
      nr_lambda_packets_for_correlation, nr_lambda_time_steps_per_packet,
      args.min_freq_channel, fpga_delays, args.fpga_ids);

  HighFive::File output_file_(args.output_filename, HighFive::File::Truncate);
  std::cout << "Creating Projection Writer" << std::endl;
  auto projection_writer = std::make_unique<HDF5ProjectionEigenWriter<
      Config::EigenvalueOutputType, Config::EigenvectorOutputType>>(
      output_file_);

  std::cout << "Creating Output Handler\n";
  auto output = std::make_shared<BufferedOutput<Config>>(
      nullptr, nullptr, std::move(projection_writer), nullptr);
  output->start_writer_loop();

  std::cout << "Initializing pipeline...\n";
  LambdaProjectionPipeline<Config, 3, 4> pipeline(num_buffers);

  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);
  pipeline.set_output(output);
  std::cout << "Initializing packet capture...\n";
  auto capture = make_packet_captures(args);
  state.nr_capture_threads = static_cast<int>(capture.size());
  INFO_LOG("Ring buffer size: {} packets\n", DEFAULT_PACKET_RING_BUFFER_SIZE);
  std::cout << "Starting threads...\n";
  std::vector<std::thread> receiver_threads;
  for (auto i = 0; i < (int)capture.size(); ++i) {
    receiver_threads.emplace_back(
        [&capture, &state, i]() { capture[i]->get_packets(state); });
  }

  std::thread processor([&state]() { state.process_packets(); });
  std::thread pipeline_feeder([&state]() { state.pipeline_feeder(); });

  // Start writer thread
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
  FLUSH_LOG();
  spdlog::shutdown();
  std::cout << "Shutdown complete.\n";
  return 0;
}
