#include "dada_def.h"
#include "spatial/common.hpp"

int main(int argc, char *argv[]) {
  std::cout << "Starting....\n";
  argparse::ArgumentParser program("pipeline");

  CommonArgs args = parse_common_args(program, argc, argv);

  std::signal(SIGINT, signal_handler);

  auto logger = setup_logger(args.debug_logging);

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
  constexpr int nr_lambda_beams = 1; // NUMBER_BEAMS;
  constexpr int nr_lambda_time_steps_per_packet = 64;
  constexpr int nr_lambda_packets_for_correlation =
      NR_OBSERVING_PACKETS_FOR_CORRELATION; // 256
  constexpr int nr_correlation_blocks_to_integrate =
      NR_OBSERVING_CORRELATION_BLOCKS_TO_INTEGRATE; // 56
  constexpr int fft_downsample_factor = 64;
  using Config = LambdaConfig<
      num_lambda_channels, nr_fpga_sources, nr_lambda_time_steps_per_packet,
      nr_lambda_receivers, nr_lambda_polarizations,
      nr_lambda_receivers_per_packet, nr_lambda_packets_for_correlation,
      nr_lambda_beams, nr_lambda_padded_receivers,
      nr_lambda_padded_receivers_per_block, nr_correlation_blocks_to_integrate,
      true, fft_downsample_factor>;

  // 2x as there will be original & RFI mitigated beams.
  using FFTOutputType =
      float[NR_OBSERVING_CHANNELS][nr_lambda_polarizations][2 * nr_lambda_beams]
           [nr_lambda_time_steps_per_packet - 10];

  using BeamOutputType =
      __half[2 * nr_lambda_beams][NR_OBSERVING_PACKETS_FOR_CORRELATION]
            [NR_OBSERVING_CHANNELS * (nr_lambda_time_steps_per_packet - 2 * 5)];
  const std::unordered_map<std::string, int> ifname_to_fpga{
      {"enp216s0np0", 3}, {"enp175s0np0", 2}, {"enp134s0np0", 1}};

  if (args.fpga_id_vec.size() != nr_fpga_sources ||
      args.fpga_ids.size() != nr_fpga_sources) {
    throw std::runtime_error("The number of network interfaces does not match "
                             "number of FPGA sources.");
  }

  auto fpga_delays = build_fpga_delay_array<nr_fpga_sources>(args);

  ProcessorState<Config, num_packet_buffers, DEFAULT_PACKET_RING_BUFFER_SIZE>
      state(
      nr_lambda_packets_for_correlation, nr_lambda_time_steps_per_packet,
      args.min_freq_channel, fpga_delays, args.fpga_ids);

  std::cout << "Creating FFT Writer" << std::endl;
  std::string filename = make_default_filename(
      "beam_fft", args.min_freq_channel, num_lambda_channels, args.fpga_id_vec);
  std::string beam_filename = make_default_filename(
      "beam", args.min_freq_channel, num_lambda_channels, args.fpga_id_vec);

  HighFive::File fft_beam_file(filename, HighFive::File::Truncate);
  //  HighFive::File beam_file(beam_filename, HighFive::File::Truncate);
  // auto fft_writer = std::make_unique<RedisBeamFFTWriter<FFTOutputType>>(
  //     Config::NR_CHANNELS, 2 * nr_lambda_beams, Config::NR_POLARIZATIONS,
  //     "beam-fft:");
  auto fft_writer = std::make_unique<HDF5BeamFFTWriter<FFTOutputType>>(
      fft_beam_file, args.min_freq_channel,
      args.min_freq_channel + num_lambda_channels - 1);

  auto beam_writer = nullptr;
  //   std::make_unique<
  // BinaryRawBeamWriter<BeamOutputType, Config::ArrivalsOutputType>>(
  // beam_filename, false);

  std::cout << "Creating Eigen Writer\n";
  std::string eigen_filename =
      make_default_filename("eigendata", args.min_freq_channel,
                            num_lambda_channels, args.fpga_id_vec);

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

  auto output =
      std::make_shared<BufferedOutput<Config, FFTOutputType, Eigenvalues,
                                      Eigenvectors, BeamOutputType>>(
          std::move(beam_writer), nullptr, std::move(eigen_writer),
          std::move(fft_writer));
  output->start_writer_loop();

  std::cout << "Loading weights...\n";
  BeamWeightsT<Config> h_weights;

  if (args.beam_weights_filename == "") {
    std::cout << "using default beam weights...\n";
    for (auto i = 0; i < num_lambda_channels; ++i) {
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
  } else {
    std::cout << "using bespoke beam weights...\n";
    for (auto i = 0; i < num_lambda_channels; ++i) {
      for (auto f = 0; f < Config::NR_FPGA_SOURCES; ++f) {
        int fpga_id = args.fpga_id_vec[f];
        for (auto k = 0; k < Config::NR_RECEIVERS_PER_PACKET; ++k) {
          for (auto j = 0; j < nr_lambda_beams; ++j) {
            for (auto l = 0; l < nr_lambda_polarizations; ++l) {
              int receiver_idx = f * Config::NR_RECEIVERS_PER_PACKET + k;
              std::string pol_string;
              if (l == 0) {
                pol_string = "XX";

              } else {
                pol_string = "YY";
              }

              h_weights.weights[i][l][j][receiver_idx] = std::complex<__half>(
                  __float2half(
                      args.beam_weights
                          ["weights"][std::to_string(args.min_freq_channel + i)]
                          [pol_string][std::to_string(
                              args.antenna_mapping[receiver_idx])]["real"]),
                  __float2half(
                      args.beam_weights
                          ["weights"][std::to_string(args.min_freq_channel + i)]
                          [pol_string][std::to_string(
                              args.antenna_mapping[receiver_idx])]["imag"]));
            }
          }
        }
      }
    }
  }

  // Fold calibration into the synthesized steering weights rather than
  // applying it again elsewhere (see compute_steering_weights() in
  // pipeline.hpp). `calibration_gains` is declared first so it outlives
  // `beam_steering`/`pipeline`.
  const bool fold_calibration_into_steering =
      !args.beam_targets.empty() && args.apply_gains;
  typename Config::AntennaGains calibration_gains{};
  if (fold_calibration_into_steering) {
    calibration_gains = get_gains_structure<Config>(args);
  }

  // LambdaPulsarFoldPipeline runs with a single GPU buffer, so num_buffers = 1.
  BeamSteering<Config> beam_steering(
      args.beam_targets, args.antenna_positions, args.antenna_mapping,
      args.frequency_plan, args.min_freq_channel, args.array_location,
      args.steering_update_interval_seconds, /*num_buffers=*/1,
      fold_calibration_into_steering ? &calibration_gains : nullptr);

  std::cout << "Initializing pipeline...\n";
  key_t rfi_dada_key = 0xbeef;
  LambdaPulsarFoldPipeline<Config, false> pipeline(
      &h_weights, args.nr_signal_eigenvectors, args.min_freq_channel,
      DADA_DEFAULT_BLOCK_KEY, "header.hdr", rfi_dada_key,
      std::move(beam_steering));

  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);
  pipeline.set_output(output);
  std::cout << "Initializing packet capture...\n";
  auto capture = make_packet_captures(args, 512 * 1024 * 1024);
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
