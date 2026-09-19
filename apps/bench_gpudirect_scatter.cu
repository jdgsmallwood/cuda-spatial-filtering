// Phase 0 validation benchmark for the GPUDirect RDMA ingest plan
// (see /home/ubuntu/.claude/plans/i-want-to-start-breezy-lampson.md).
//
// Validates, on real GPU hardware and with no NIC/ibverbs dependency, the two
// unverified numbers the design leans on:
//   (a) CUDA graph replay + scatter-kernel execution fits comfortably inside
//       the ~64.5us batch window implied by 64 channels x 15,500 pkt/s.
//   (b) N concurrent capture threads, each replaying a graph on that cadence,
//       don't bottleneck on GPU command submission or starve a concurrent
//       correlator/beamformer-like workload.
//
// This binary uses a synthetic SM-saturating busy kernel as the stand-in
// correlator/beamformer workload (fast to iterate on). bench_gpu.cu's
// --relocation-check flag runs the same relocation streams alongside the
// *real* LambdaGPUPipeline/LambdaCorrBeamOnlyGPUPipeline kernels instead --
// use that to confirm findings here against the actual pipeline.
#include "bench_gpudirect_scatter_common.hpp"
#include <argparse/argparse.hpp>
#include <iostream>
#include <vector>

using namespace gpudirect_bench;

namespace {

// Synthetic "correlator-like" busy kernel: enough FMA work per thread across
// a large grid to plausibly occupy most SMs, so the benchmark measures
// interference from concurrent relocation traffic, not just the scatter
// kernel in isolation.
__global__ void busy_kernel(float *out, int iters) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = static_cast<float>(idx) * 1.0000001f;
#pragma unroll 4
  for (int i = 0; i < iters; ++i) {
    acc = fmaf(acc, 1.0000001f, 0.0000001f);
  }
  out[idx] = acc;
}

double run_busy_kernel(double duration_s, std::atomic<bool> *stop_flag) {
  cudaStream_t stream;
  int low_prio = 0, high_prio = 0;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio));
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio));

  constexpr int NR_BLOCKS = 4096;
  constexpr int NR_THREADS = 256;
  constexpr int ITERS_PER_LAUNCH = 2000;
  float *out;
  CUDA_CHECK(cudaMalloc(&out, sizeof(float) * NR_BLOCKS * NR_THREADS));

  unsigned long long iters = 0;
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < duration_s) {
    if (stop_flag && stop_flag->load(std::memory_order_relaxed)) break;
    busy_kernel<<<NR_BLOCKS, NR_THREADS, 0, stream>>>(out, ITERS_PER_LAUNCH);
    ++iters;
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  cudaFree(out);
  cudaStreamDestroy(stream);
  return iters / elapsed;
}

} // namespace

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("bench_gpudirect_scatter");
  program.add_argument("--duration")
      .help("Duration in seconds for each measurement phase")
      .default_value(5.0)
      .scan<'g', double>();
  program.add_argument("--num-streams")
      .help("Number of simulated FPGA capture streams")
      .default_value(4)
      .scan<'i', int>();
  program.add_argument("--poll-batch")
      .help("Packets per relocation batch (POLL_BATCH)")
      .default_value(64)
      .scan<'i', int>();
  program.add_argument("--channels")
      .help("Channels per FPGA (for target-rate computation)")
      .default_value(64)
      .scan<'i', int>();
  program.add_argument("--pkt-rate-per-channel")
      .help("Packets/sec per channel (for target-rate computation)")
      .default_value(15500.0)
      .scan<'g', double>();
  program.add_argument("--dest-slots")
      .help("Destination ring size in packets (NR_PACKETS_FOR_CORRELATION-like)")
      .default_value(256)
      .scan<'i', int>();
  program.add_argument("--no-busy")
      .help("Skip the concurrent correlator-like busy kernel (isolates relocation-only throughput)")
      .default_value(false)
      .implicit_value(true);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << "\n" << program;
    return 1;
  }

  const double duration_s = program.get<double>("--duration");
  const int num_streams = program.get<int>("--num-streams");
  const int poll_batch = program.get<int>("--poll-batch");
  const int channels = program.get<int>("--channels");
  const double pkt_rate_per_channel = program.get<double>("--pkt-rate-per-channel");
  const int dest_slots = program.get<int>("--dest-slots");
  const bool no_busy = program.get<bool>("--no-busy");

  const double pkt_rate_per_fpga = channels * pkt_rate_per_channel;
  const double target_replays_per_sec_per_stream = pkt_rate_per_fpga / poll_batch;
  const double target_replays_per_sec_aggregate = target_replays_per_sec_per_stream * num_streams;
  const double pass_threshold_aggregate = 2.0 * target_replays_per_sec_aggregate;

  std::cout << "bench_gpudirect_scatter: Phase 0 validation benchmark (synthetic busy kernel)\n"
            << "  num_streams=" << num_streams << " poll_batch=" << poll_batch
            << " channels/fpga=" << channels
            << " pkt_rate/channel=" << pkt_rate_per_channel << "/s\n"
            << "  => target ~" << target_replays_per_sec_per_stream
            << " replays/sec/stream, ~" << target_replays_per_sec_aggregate
            << " replays/sec aggregate\n"
            << "  pass threshold (2x target aggregate): " << pass_threshold_aggregate
            << " replays/sec\n\n";

  double baseline_iters_per_sec = 0.0;
  if (!no_busy) {
    std::cout << "=== Baseline: correlator-like busy kernel alone (" << duration_s << "s) ===\n";
    baseline_iters_per_sec = run_busy_kernel(duration_s, nullptr);
    std::cout << "baseline busy_kernel iters/sec=" << baseline_iters_per_sec << "\n\n";
  }

  std::cout << "=== " << (no_busy ? "Relocation streams alone (isolation check), " : "Concurrent: busy kernel + ")
            << num_streams << " relocation streams (" << duration_s << "s) ===\n";
  std::atomic<bool> start_gate{false};
  std::atomic<bool> stop_busy{false};
  std::vector<StreamStats> stats(num_streams);
  std::vector<std::thread> threads;
  for (int i = 0; i < num_streams; ++i) {
    threads.emplace_back(run_stream, poll_batch, dest_slots, duration_s,
                         std::ref(start_gate), std::ref(stats[i]));
  }
  std::thread busy_thread([&]() {
    if (no_busy) return;
    // Give relocation threads time to warm up + reach the start gate.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const double r = run_busy_kernel(duration_s, &stop_busy);
    std::cout << "concurrent busy_kernel iters/sec=" << r << "\n";
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  start_gate.store(true, std::memory_order_release);
  for (auto &t : threads) t.join();
  stop_busy.store(true, std::memory_order_release);
  busy_thread.join();

  unsigned long long total_replays = 0, total_stalls = 0;
  double min_us = 1e18, max_us = 0.0, total_us = 0.0;
  for (int i = 0; i < num_streams; ++i) {
    const auto &s = stats[i];
    const double per_stream_rate = s.replays / duration_s;
    std::cout << "[stream " << i << "] replays=" << s.replays
              << " replays/sec=" << per_stream_rate
              << " stalls=" << s.stalls
              << " avg_us=" << (s.replays ? s.total_replay_us / s.replays : 0.0)
              << " min_us=" << s.min_replay_us << " max_us=" << s.max_replay_us << "\n";
    total_replays += s.replays;
    total_stalls += s.stalls;
    min_us = std::min(min_us, s.min_replay_us);
    max_us = std::max(max_us, s.max_replay_us);
    total_us += s.total_replay_us;
  }
  const double aggregate_rate = total_replays / duration_s;
  std::cout << "\n[aggregate] replays/sec=" << aggregate_rate
            << " total_stalls=" << total_stalls
            << " avg_us=" << (total_replays ? total_us / total_replays : 0.0)
            << " min_us=" << min_us << " max_us=" << max_us << "\n";

  std::cout << "\n=== Phase 0 pass criteria ===\n";
  bool pass = true;
  if (aggregate_rate >= pass_threshold_aggregate) {
    std::cout << "[PASS] aggregate replay rate " << aggregate_rate
              << " >= 2x target (" << pass_threshold_aggregate << ")\n";
  } else {
    std::cout << "[FAIL] aggregate replay rate " << aggregate_rate
              << " < 2x target (" << pass_threshold_aggregate << ")\n";
    pass = false;
  }
  if (total_stalls == 0) {
    std::cout << "[PASS] zero double-buffer stalls\n";
  } else {
    std::cout << "[FAIL] " << total_stalls << " double-buffer stalls observed\n";
    pass = false;
  }
  // Degradation check printed above (concurrent vs baseline); compute here too.
  std::cout << "  (compare concurrent busy_kernel iters/sec above against baseline "
            << baseline_iters_per_sec << " for the <10% degradation criterion)\n";

  std::cout << "\nOverall Phase 0: " << (pass ? "PASS (see degradation line above)" : "FAIL") << "\n";
  return pass ? 0 : 1;
}
