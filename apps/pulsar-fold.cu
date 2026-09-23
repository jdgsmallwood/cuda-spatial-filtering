#include "dada_def.h"
#include "spatial/common.hpp"
#include "spatial/deripple.hpp"
#include <optional>

int main(int argc, char *argv[]) {
  std::cout << "Starting....\n";
  argparse::ArgumentParser program("pipeline");
  CommonArgs args = parse_common_args(program, argc, argv);

  std::signal(SIGINT, signal_handler);

  auto logger = setup_logger(args.debug_logging);

  constexpr int nr_fpga_sources = NR_OBSERVING_FPGA_SOURCES;
  constexpr size_t num_packet_buffers = 24;
  // Four ibverbs QPs keep about 4096 receive slots posted. At 40 channels,
  // deferred packets can occupy several thousand more slots before their
  // assembly windows become available. Keep headroom for both populations.
  constexpr size_t packet_ring_size = 32768;
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
      true, fft_downsample_factor, NR_OBSERVING_FINE_CHANNELS,
      NR_OBSERVING_FINE_CHANNEL_EDGE_TRIM>;

  std::optional<CoarsePfbDeripple> deripple;
  if (!args.deripple_config_filename.empty()) {
    deripple = load_coarse_pfb_deripple(
        args.deripple_config_filename, Config::NR_FINE_CHANNELS,
        Config::NR_FINE_CHANNEL_EDGE_TRIM);
    std::cout << "Using pulsar-output de-ripple response "
              << deripple->response_id << "\n";
  }

  if (args.fpga_id_vec.size() != nr_fpga_sources ||
      args.fpga_ids.size() != nr_fpga_sources) {
    throw std::runtime_error("The number of network interfaces does not match "
                             "number of FPGA sources.");
  }

  auto fpga_delays = build_fpga_delay_array<nr_fpga_sources>(args);

  ProcessorState<Config, num_packet_buffers, packet_ring_size>
      state(
      nr_lambda_packets_for_correlation, nr_lambda_time_steps_per_packet,
      args.min_freq_channel, fpga_delays, args.fpga_ids);

  // The only production output of this app is the PSRDADA ring.  Keep the
  // Output object non-null (the pipeline uses that as its output-enable bit),
  // but do not create the old diagnostic FFT/eigen HDF5 writers: at 1120 fine
  // channels they consume substantial memory and I/O without helping DSPSR.
  std::string audit_filename = make_default_filename(
      "pulsar_fold", args.min_freq_channel, num_lambda_channels,
      args.fpga_id_vec);
  write_stream_mapping_csv(
      args, audit_sidecar_filename(audit_filename),
      nr_lambda_receivers_per_packet, nr_lambda_polarizations);
  auto output = std::make_shared<BufferedOutput<Config>>(
      nullptr, nullptr, nullptr, nullptr);

  std::cout << "Loading weights...\n";
  BeamWeightsT<Config> h_weights{};

  if (args.beam_weights_filename == "") {
    std::cout << "using default beam weights...\n";
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
  } else {
    std::cout << "using bespoke beam weights...\n";
    for (auto i = 0; i < Config::NR_CHANNELS; ++i) {
      const auto coarse_i = i / Config::NR_EFFECTIVE_FINE_CHANNELS;
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
                          ["weights"][std::to_string(args.min_freq_channel + coarse_i)]
                          [pol_string][std::to_string(
                              args.antenna_mapping[receiver_idx])]["real"]),
                  __float2half(
                      args.beam_weights
                          ["weights"][std::to_string(args.min_freq_channel + coarse_i)]
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
  const bool use_canonical = !args.canonical_recv_perm.empty();
  const auto &active_mapping =
      use_canonical ? args.canonical_antenna_mapping : args.antenna_mapping;
  auto beam_mapping = active_mapping;
  for (int excluded_antenna : args.beam_excluded_antennas) {
    bool found = false;
    for (auto &[receiver_idx, antenna_id] : beam_mapping) {
      if (antenna_id == excluded_antenna) {
        antenna_id = -1;
        found = true;
        std::cout << "Excluding physical antenna " << excluded_antenna
                  << " from coherent beam (receiver " << receiver_idx << ")\n";
      }
    }
    if (!found)
      throw std::runtime_error("Requested beam exclusion antenna " +
                               std::to_string(excluded_antenna) +
                               " is absent from the active stream map");
  }
  typename Config::FineAntennaGains calibration_gains{};
  if (fold_calibration_into_steering) {
    calibration_gains = get_fine_beam_gains_structure<Config>(
        args, beam_mapping);
  }

  // The DADA sink rotates three GPU buffers to overlap host transfer,
  // channelization/beamforming, and DADA output transfer.
  BeamSteering<Config> beam_steering(
      args.beam_targets, args.antenna_positions, beam_mapping,
      args.frequency_plan, args.min_freq_channel, args.array_location,
      args.steering_update_interval_seconds, /*num_buffers=*/3,
      fold_calibration_into_steering ? &calibration_gains : nullptr,
      args.steering_utc);

  std::cout << "Initializing pipeline...\n";
  key_t rfi_dada_key = 0xbeef;
#ifdef PULSAR_DADA_DETECTED_INTENSITY
  LambdaPulsarFoldPipeline<Config, false, true> pipeline(
#else
  LambdaPulsarFoldPipeline<Config, false, false> pipeline(
#endif
      &h_weights, args.nr_signal_eigenvectors, args.min_freq_channel,
      DADA_DEFAULT_BLOCK_KEY, args.dada_header_filename, rfi_dada_key,
      std::move(beam_steering), deripple ? &*deripple : nullptr);

  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);
  pipeline.set_output(output);
  if (use_canonical)
    pipeline.set_stream_permutation(args.canonical_recv_perm, args.canonical_pol_perm);

  // The DADA header is published from the pipeline constructor, before its
  // CUDA warm-up. DSPSR may still need substantially longer than that warm-up
  // to build its coherent-dedispersion plan on the shared GPU. The launcher
  // creates this sentinel only after DSPSR reports "prepared in"; do not open
  // capture sockets until then or finite buffers merely hide a startup overrun.
  if (const char *ready_file = std::getenv("SPATIAL_CAPTURE_READY_FILE")) {
    std::cout << "Waiting for downstream-ready sentinel: " << ready_file
              << std::endl;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(120);
    while (!std::ifstream(ready_file).good() &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!std::ifstream(ready_file).good())
      throw std::runtime_error(
          "timed out waiting for downstream consumer readiness");
    std::cout << "Downstream consumer is prepared; capture may start."
              << std::endl;
  }
  std::cout << "Initializing packet capture...\n";
  auto capture = make_packet_captures(args, 512 * 1024 * 1024);
  // The ibverbs direct-to-ring arming step reserves its initial WR slots via
  // reserve_write_batch_strided().  Publish the producer count first so each
  // QP gets a disjoint lane (i, i+N, i+2N, ...).  Arming while this is still
  // zero makes every QP use stride one and overlap another QP's reservations,
  // pairing one FPGA's payload with another FPGA's source-IP metadata.
  // PCAPPacketCapture uses the legacy sequential write_index path. Only live
  // NIC captures publish the per-thread claims consumed by strided mode.
  state.nr_capture_threads = args.pcap_filename.empty()
                                 ? static_cast<int>(capture.size())
                                 : 0;
#ifdef HAVE_IBVERBS
  // Keep capture ownership consistent with ProcessorState's four-producer
  // strided consumer.  Without arming this path, LibibverbsPacketCapture
  // falls back to its legacy shared sequential writer while process_packets()
  // consumes per-thread strided claims; the 6144-slot ring then fills with
  // Received=6144, Processed=0 and cannot make forward progress.
  if (args.pcap_filename.empty() && args.capture_backend == "ibverbs") {
    std::cout << "ibverbs receive mode: direct into packet ring (no staging memcpy)\n";
    arm_ibverbs_zero_copy_captures(capture, state);
  }
#endif
  INFO_LOG("Ring buffer size: {} packets\n", packet_ring_size);
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
  FLUSH_LOG();
  spdlog::shutdown();
  std::cout << "Shutdown complete.\n";
  return 0;
}
