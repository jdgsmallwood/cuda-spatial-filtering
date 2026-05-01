#include "spatial/common.hpp"

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
  constexpr int nr_lambda_beams = 1; // NUMBER_BEAMS;
  constexpr int nr_lambda_time_steps_per_packet = 64;
  constexpr int nr_lambda_packets_for_correlation =
      NR_OBSERVING_PACKETS_FOR_CORRELATION; // 256
  constexpr int nr_correlation_blocks_to_integrate =
      NR_OBSERVING_CORRELATION_BLOCKS_TO_INTEGRATE; // 56
  constexpr int fft_downsample_factor = 64;
  constexpr size_t PACKET_RING_BUFFER_SIZE = 50000;
  using Config = LambdaConfig<
      num_lambda_channels, nr_fpga_sources, nr_lambda_time_steps_per_packet,
      nr_lambda_receivers, nr_lambda_polarizations,
      nr_lambda_receivers_per_packet, nr_lambda_packets_for_correlation,
      nr_lambda_beams, nr_lambda_padded_receivers, nr_lambda_padded_receivers,
      nr_correlation_blocks_to_integrate, true, fft_downsample_factor>;

  // 2x as there will be original & RFI mitigated beams.
  using FFTOutputType =
      float[NR_OBSERVING_CHANNELS][nr_lambda_polarizations][2 * nr_lambda_beams]
           [nr_lambda_time_steps_per_packet *
            NR_OBSERVING_PACKETS_FOR_CORRELATION / fft_downsample_factor];

  using BeamOutputType =
      __half[2 * nr_lambda_beams][NR_OBSERVING_PACKETS_FOR_CORRELATION]
            [NR_OBSERVING_CHANNELS * (nr_lambda_time_steps_per_packet - 2 * 5)];
  const std::unordered_map<std::string, int> ifname_to_fpga{
      {"enp216s0np0", 3}, {"enp175s0np0", 2}, {"enp134s0np0", 1}};

  using MapType = std::unordered_map<uint32_t, int>;
  auto fpga_ids = std::make_unique<MapType>();
  std::vector<int> fpga_id_vec;
  auto fpga_names = split_ifnames(args.ifname);

  {
    // use scope here to deallocate i at the end.
    int i = 0;
    for (const auto &name : fpga_names) {
      int fpga_id = 0;

      auto it = ifname_to_fpga.find(name);
      if (it != ifname_to_fpga.end()) {
        fpga_id = it->second;
      }
      (*fpga_ids)[fpga_id] = i;
      fpga_id_vec.push_back(fpga_id);
      i++;
    }
  }

  if (fpga_id_vec.size() != nr_fpga_sources ||
      fpga_ids->size() != nr_fpga_sources) {
    throw std::runtime_error("The number of network interfaces does not match "
                             "number of FPGA sources.");
  }

  std::array<int64_t, nr_fpga_sources> fpga_delays;
  for (auto i = 0; i < nr_fpga_sources; ++i) {
    fpga_delays[i] = 0;
  }

  if (nr_fpga_sources == 2) {
    fpga_delays[1] = args.fpga_delay;
  }

  ProcessorState<Config, num_packet_buffers, PACKET_RING_BUFFER_SIZE> state(
      nr_lambda_packets_for_correlation, nr_lambda_time_steps_per_packet,
      args.min_freq_channel, fpga_delays, &fpga_ids);

  AntennaMapRegistry registry;

  std::unordered_map<int, int> antenna_mapping =
      registry.get_combined_map(fpga_id_vec);
  std::cout << "Antenna mapping is:\n";
  for (const auto &[key, val] : antenna_mapping) {
    std::cout << "Key: " << key << ", Val: " << val << std::endl;
  };

  std::cout << "Creating FFT Writer" << std::endl;
  std::string filename = make_default_filename(
      "beam_fft", args.min_freq_channel, num_lambda_channels, fpga_id_vec);
  std::string beam_filename = make_default_filename(
      "beam", args.min_freq_channel, num_lambda_channels, fpga_id_vec);

  // HighFive::File fft_beam_file(filename, HighFive::File::Truncate);
  // //  HighFive::File beam_file(beam_filename, HighFive::File::Truncate);
  // // auto fft_writer = std::make_unique<RedisBeamFFTWriter<FFTOutputType>>(
  // //     Config::NR_CHANNELS, 2 * nr_lambda_beams, Config::NR_POLARIZATIONS,
  // //     "beam-fft:");
  // auto fft_writer = std::make_unique<HDF5BeamFFTWriter<FFTOutputType>>(
  //     fft_beam_file, min_freq_channel,
  //     min_freq_channel + num_lambda_channels - 1);

  auto beam_writer = std::make_unique<
      BinaryRawBeamWriter<BeamOutputType, Config::ArrivalsOutputType>>(
      beam_filename, false);

  std::cout << "Creating Eigen Writer\n";
  std::string eigen_filename = make_default_filename(
      "eigendata", args.min_freq_channel, num_lambda_channels, fpga_id_vec);

  HighFive::File eigendata_file(eigen_filename, HighFive::File::Truncate);
  // auto fft_writer = std::make_unique<RedisBeamFFTWriter<FFTOutputType>>(
  //     Config::NR_CHANNELS, 2 * nr_lambda_beams, Config::NR_POLARIZATIONS,
  //     "beam-fft:");
  //
  using Eigenvalues =
      float[Config::NR_CHANNELS][Config::NR_POLARIZATIONS][nr_lambda_receivers];
  using Eigenvectors =
      std::complex<float>[Config::NR_CHANNELS][Config::NR_POLARIZATIONS]
                         [nr_lambda_receivers][nr_lambda_receivers];
  auto eigen_writer =
      std::make_unique<HDF5EigenWriter<Eigenvalues, Eigenvectors>>(
          eigendata_file);

  std::cout << "Creating Output Handler\n";

  auto output = std::make_shared<
      BufferedOutput<Config, FFTOutputType, Eigenvalues, Eigenvectors,
                     Config::PulsarFoldOutputType, BeamOutputType>>(
      std::move(beam_writer), nullptr, std::move(eigen_writer),
      nullptr /*std::move(fft_writer) */, nullptr, 100, 100, 100, 200, 100);

  std::cout << "Loading weights...\n";
  BeamWeightsT<Config> h_weights;

  for (auto i = 0; i < num_lambda_channels; ++i) {
    for (auto j = 0; j < nr_lambda_receivers; ++j) {
      for (auto k = 0; k < nr_lambda_beams; ++k) {
        for (auto l = 0; l < nr_lambda_polarizations; ++l) {
          h_weights.weights[i][l][k][j] =
              std::complex<__half>(__float2half(1.0f), __float2half(0.0f));
        }
      }
    }
  }

  std::cout << "Initializing pipeline...\n";
  LambdaAdaptiveBeamformedSpectraPipeline<Config> pipeline(num_buffers,
                                                           &h_weights);

  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);
  pipeline.set_output(output);
  std::cout << "Initializing packet capture...\n";
  std::vector<std::unique_ptr<PacketInput>> capture;

  if (!args.pcap_filename.empty()) {
    capture.push_back(std::make_unique<PCAPPacketCapture>(args.pcap_filename,
                                                          args.loop_pcap));
  } else {
    for (auto nic : fpga_names) {
      capture.push_back(std::make_unique<KernelSocketPacketCapture>(
          nic, args.port, BUFFER_SIZE, 256 * 1024 * 1024));
    }
  }
  LOG_INFO("Ring buffer size: {} packets\n", PACKET_RING_BUFFER_SIZE);
  std::cout << "Starting threads...\n";
  std::vector<std::thread> receiver_threads;
  for (auto i = 0; i < capture.size(); ++i) {
    receiver_threads.emplace_back(
        [&capture, &state, i]() { capture[i]->get_packets(state); });
  }

  std::thread processor([&state]() { state.process_packets(); });
  std::thread pipeline_feeder([&state]() { state.pipeline_feeder(); });

  // Start writer thread
  std::thread writer_thread_([&output] { output->writer_loop(); });
  std::cout << "Setup completed. Ready to receive!" << std::endl;
  // Print statistics periodically
  int packets_received = 0;
  int timeout = 0;
  while (state.running) {
    sleep(5);
    // This is nice to see outside of log files.
    std::cout << "Stats: Received=" << state.packets_received
              << ", Processed=" << state.packets_processed
              << ", Missing=" << state.packets_missing
              << ", Discarded=" << state.packets_discarded << std::endl;
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
  }

  // Cleanup
  LOG_INFO("\nShutting down...\n");
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

  output->running_ = false;
  std::cout << "Waiting for writer thread to finish...\n";
  writer_thread_.join();
  FLUSH_LOG();
  spdlog::shutdown();
  std::cout << "Shutdown complete.\n";
  return 0;
}
