// Correlation-only observing app: ingest -> delay/reorder -> pre-correlation fine
// channelization -> TCC correlation -> visibilities, with no beamforming, no
// eigendecomposition/RFI mitigation, and no spectral (FFT) output. See
// LambdaStarweavePipeline (include/spatial/pipeline/lambda_starweave_pipeline.hpp).
//
// Modeled on apps/get_projection_matrix.cu's lean structure (no BeamSteering/timing-CSV
// machinery), adopting apps/observe.cu's visibilities-file setup (audit sidecar,
// HDF5VisibilitiesWriter). Capture backend selection (kernel/ibverbs/ibverbs-gpudirect) is
// entirely generic via make_packet_captures()/--capture-backend; arm_gpudirect_captures()
// wires up the GPUDirect zero-copy path when built with libibverbs.
#include "spatial/common.hpp"

#ifndef NR_OBSERVING_FINE_CHANNELS
#define NR_OBSERVING_FINE_CHANNELS 1
#endif

#ifndef NR_OBSERVING_FINE_CHANNEL_EDGE_TRIM
#define NR_OBSERVING_FINE_CHANNEL_EDGE_TRIM 0
#endif

#ifndef NR_OBSERVING_PACKET_BUFFERS
#define NR_OBSERVING_PACKET_BUFFERS 8
#endif

int main(int argc, char *argv[]) {
  std::cout << "Starting....\n";
  argparse::ArgumentParser program("pipeline");
  CommonArgs args = parse_common_args(program, argc, argv);

  std::signal(SIGINT, signal_handler);

  auto logger = setup_logger(args.debug_logging);
  constexpr int num_buffers = NR_OBSERVING_BUFFERS;
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
  // No beamforming in this pipeline -- NR_BEAMS is unused outside beam-specific
  // LambdaConfig-derived types, so a placeholder of 1 is inert.
  constexpr int nr_lambda_beams = 1;
  constexpr int nr_lambda_time_steps_per_packet = 64;
  constexpr int nr_lambda_packets_for_correlation =
      NR_OBSERVING_PACKETS_FOR_CORRELATION; // 256
  constexpr int nr_correlation_blocks_to_integrate =
      NR_OBSERVING_CORRELATION_BLOCKS_TO_INTEGRATE; // 56
  constexpr int num_lambda_fine_channels = NR_OBSERVING_FINE_CHANNELS;
  constexpr int num_lambda_fine_channel_edge_trim =
      NR_OBSERVING_FINE_CHANNEL_EDGE_TRIM;
  using Config =
      LambdaConfig<num_lambda_channels, nr_fpga_sources,
                   nr_lambda_time_steps_per_packet, nr_lambda_receivers,
                   nr_lambda_polarizations, nr_lambda_receivers_per_packet,
                   nr_lambda_packets_for_correlation, nr_lambda_beams,
                   nr_lambda_padded_receivers,
                   nr_lambda_padded_receivers_per_block,
                   nr_correlation_blocks_to_integrate, true, 256,
                   num_lambda_fine_channels, num_lambda_fine_channel_edge_trim>;

  if (args.fpga_id_vec.size() != nr_fpga_sources ||
      args.fpga_ids.size() != nr_fpga_sources) {
    throw std::runtime_error("The number of network interfaces does not match "
                             "number of FPGA sources.");
  }
  auto fpga_delays = build_fpga_delay_array<nr_fpga_sources>(args, true);

  auto gains = get_gains_structure<Config>(args);
  const bool use_canonical = !args.canonical_recv_perm.empty();
  const auto &active_mapping =
      use_canonical ? args.canonical_antenna_mapping : args.antenna_mapping;
  ProcessorState<Config, num_packet_buffers, DEFAULT_PACKET_RING_BUFFER_SIZE,
                 NR_OBSERVING_PACKET_WORKER_THREADS>
      state(
      nr_lambda_packets_for_correlation, nr_lambda_time_steps_per_packet,
      args.min_freq_channel, fpga_delays, args.fpga_ids);

  if (!program.is_used("-v")) {
    args.output_filename =
        make_default_filename("visibilities", args.min_freq_channel,
                              num_lambda_channels, args.fpga_id_vec);
  }
  write_stream_mapping_csv(
      args, audit_sidecar_filename(args.output_filename),
      nr_lambda_receivers_per_packet, nr_lambda_polarizations);
  HighFive::File vis_file(args.output_filename, HighFive::File::Truncate);
  write_hdf5_run_audit(
      vis_file, args, argc, argv, nr_lambda_receivers_per_packet,
      nr_lambda_polarizations);

  auto vis_writer =
      std::make_unique<HDF5VisibilitiesWriter<Config::VisibilitiesOutputType>>(
          vis_file, args.min_freq_channel,
          // HDF5 min/max describe FPGA coarse channels; the visibility
          // dataset's channel axis contains the post-PFB fine bins.
          args.min_freq_channel + Config::NR_FPGA_CHANNELS - 1,
          &active_mapping, 100, 0, use_canonical);

  // No beam/eigen/FFT output -- BufferedOutput tolerates null writers for the
  // slots this pipeline never registers blocks against.
  auto output = std::make_shared<BufferedOutput<Config>>(
      nullptr, std::move(vis_writer), nullptr, nullptr);

  std::cout << "Initializing pipeline...\n";
  LambdaStarweavePipeline<Config> pipeline(num_buffers,
                                           nr_correlation_blocks_to_integrate);

  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);
  pipeline.set_output(output);
  if (use_canonical)
    pipeline.set_stream_permutation(args.canonical_recv_perm, args.canonical_pol_perm);
  if (args.apply_gains) {
    std::cout << "Applying gains as -a is selected!" << std::endl;
    pipeline.set_antenna_gains((std::complex<float> *)gains.data());
  } else {
    std::cout << "Not applying gains as -a is not selected" << std::endl;
  }

  std::cout << "Initializing packet capture...\n";
  auto capture = make_packet_captures(args);
#ifdef HAVE_IBVERBS
  // No-op for every backend except ibverbs-gpudirect -- must run after the pipeline's GPU
  // landing buffers exist (they do, pipeline is constructed above) and before receiver
  // threads start (they haven't yet).
  arm_gpudirect_captures(capture, state);
#endif
  if (args.pcap_filename.empty())
    state.nr_capture_threads = static_cast<int>(capture.size());
#ifdef HAVE_IBVERBS
  if (args.capture_backend == "ibverbs") {
    std::cout << "ibverbs receive mode: direct into packet ring (no staging memcpy)\n";
    arm_ibverbs_zero_copy_captures(capture, state);
  }
#endif
  INFO_LOG("Ring buffer size: {} packets\n", DEFAULT_PACKET_RING_BUFFER_SIZE);
  std::cout << "Starting threads....\n";
  std::vector<std::thread> receiver_threads;
  for (auto i = 0; i < (int)capture.size(); ++i) {
    receiver_threads.emplace_back(
        [&capture, &state, i]() { capture[i]->get_packets(state); });
  }

  std::thread processor([&state]() { state.process_packets(); });
  std::thread pipeline_feeder([&state]() { state.pipeline_feeder(); });

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

  std::cout << "Stopping writers...\n";
  output->running_ = false;
  output->stop_writers();
  FLUSH_LOG();
  spdlog::shutdown();
  std::cout << "Shutdown complete.\n";
  return 0;
}
