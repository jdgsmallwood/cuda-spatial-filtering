#include "bench_gpudirect_scatter_common.hpp"
#include "spatial/logging.hpp"
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.hpp"
#include <argparse/argparse.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <thread>
#include <vector>

// Fixed representative GPU pipeline configs -- 8..32 channels in steps of 8
// for the production 4-FPGA (40 rx) layout. Each config is a separate
// template instantiation so a single binary benchmarks all shapes without
// needing different -DNR_OBSERVING_* CMake builds.
//
// NR_PADDED_RECEIVERS must be the next multiple of 32 ≥ NR_RECEIVERS:
//   40 rx  → 64 padded   (4 fpga)
//
//                                     ch  fp   ts   rx  pol rxpp corr  bm  pad  blk       acc
using Cfg8ch4fpga  = LambdaConfig<     8,  4,   64,  40,  2,  10, 256,  1,  64,  32, 10000000>;
using Cfg16ch4fpga = LambdaConfig<    16,  4,   64,  40,  2,  10, 256,  1,  64,  32, 10000000>;
using Cfg24ch4fpga = LambdaConfig<    24,  4,   64,  40,  2,  10, 256,  1,  64,  32, 10000000>;
using Cfg32ch4fpga = LambdaConfig<    32,  4,   64,  40,  2,  10, 256,  1,  64,  32, 10000000>;

// Fine-channelization comparison: same coarse-channel/FPGA/receiver shape as
// the configs above, but with gpu-filter's pre-correlation PFB active at the
// repo's default settings (NR_OBSERVING_FINE_CHANNELS=32,
// NR_OBSERVING_FINE_CHANNEL_EDGE_TRIM=2 -- see CMakeLists.txt/CLAUDE.md).
// Trailing template args: OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET=false,
// FFT_DOWNSAMPLE_FACTOR=128 (default), NR_FINE_CHANNELS=32,
// NR_FINE_CHANNEL_EDGE_TRIM=2. Effective (post-trim) channel count is
// NR_FPGA_CHANNELS * (32 - 2*2) = NR_FPGA_CHANNELS * 28, e.g. 8 coarse ->
// 224 effective -- substantially more correlator/beamformer work than the
// same coarse-channel count with fine channelization off, so only the
// smallest configs are instantiated here to stay within this dev GPU's VRAM.
//                                            ch  fp   ts   rx  pol rxpp corr  bm  pad  blk       acc     oct  fft  fine trim
using Cfg8ch4fpga_fine32  = LambdaConfig<      8,  4,   64,  40,  2,  10, 256,  1,  64,  32, 10000000, false, 128,  32,   2>;
using Cfg16ch4fpga_fine32 = LambdaConfig<     16,  4,   64,  40,  2,  10, 256,  1,  64,  32, 10000000, false, 128,  32,   2>;

// Correlation-packet sweep: 8ch/4fpga held fixed while
// NR_PACKETS_FOR_CORRELATION varies 64→1024 (powers of 2). The 256-packet
// case reuses the config above.
//                                       ch  fp   ts   rx  pol rxpp  corr   bm  pad  blk       acc
using Cfg8ch4fpga_c64   = LambdaConfig<  8,  4,   64,  40,  2,  10,   64,  1,  64,  32, 10000000>;
using Cfg8ch4fpga_c128  = LambdaConfig<  8,  4,   64,  40,  2,  10,  128,  1,  64,  32, 10000000>;
using Cfg8ch4fpga_c512  = LambdaConfig<  8,  4,   64,  40,  2,  10,  512,  1,  64,  32, 10000000>;
using Cfg8ch4fpga_c1024 = LambdaConfig<  8,  4,   64,  40,  2,  10, 1024,  1,  64,  32, 10000000>;

// --------------------------------------------------------------------------
// Minimal FinalPacketData and ProcessorState stubs -- identical in purpose
// to the ones in gpu_benchmark.cu; reproduced here as templates so they work
// across all Config specialisations without modification.
// --------------------------------------------------------------------------

template <typename T>
struct DummyFinalPacketData : public FinalPacketData {
  using sampleT  = typename T::InputPacketSamplesType;
  using scaleT   = typename T::PacketScalesType;
  using arrivalsT = typename T::ArrivalsOutputType;

  sampleT  *samples;
  scaleT   *scales;
  bool     *arrivals;

  DummyFinalPacketData() {
    CUDA_CHECK(cudaMallocHost((void **)&samples,  sizeof(sampleT)));
    CUDA_CHECK(cudaMallocHost((void **)&scales,   sizeof(scaleT)));
    CUDA_CHECK(cudaMallocHost((void **)&arrivals, sizeof(arrivalsT)));
    std::memset(samples,  1, sizeof(sampleT));
    std::memset(scales,   0, sizeof(scaleT));
    std::memset(arrivals, 0, sizeof(arrivalsT));
  }
  ~DummyFinalPacketData() {
    cudaFreeHost(samples);
    cudaFreeHost(scales);
    cudaFreeHost(arrivals);
  }

  void   *get_samples_ptr()            override { return samples; }
  size_t  get_samples_elements_size()  override { return sizeof(sampleT); }
  void   *get_scales_ptr()             override { return scales; }
  size_t  get_scales_element_size()    override { return sizeof(scaleT); }
  bool   *get_arrivals_ptr()           override { return arrivals; }
  size_t  get_arrivals_size()          override { return sizeof(arrivalsT); }
  void    zero_missing_packets()       override {}
  int     get_num_missing_packets()    override { return 0; }
};

struct FakeProcessorState : public ProcessorStateBase {
  void release_buffer(const int)                                   override {}
  void *get_next_write_pointer()                                   override { return nullptr; }
  void *get_current_write_pointer()                                override { return nullptr; }
  void add_received_packet_metadata(const int, const sockaddr_in &) override {}
  int  reserve_write_batch(int, void **, int *)                    override { return 0; }
  void commit_write_batch(int, const int *, const int *,
                          const sockaddr_in *)                     override {}
  void set_pipeline(GPUPipeline *)                                 override {}
  void process_all_available_packets()                             override {}
  void handle_buffer_completion(bool)                              override {}
};

// --------------------------------------------------------------------------
// Per-config GPU benchmark
// --------------------------------------------------------------------------

struct GpuBenchResult {
  size_t nr_channels;
  size_t nr_fpga_sources;
  size_t nr_receivers;
  unsigned long long runs;
  double elapsed;
  double runs_per_sec;
  size_t input_bytes;
  size_t output_bytes;
  double input_gb_per_sec;
  double output_gb_per_sec;
  double total_gb_per_sec;
};

template <typename T>
GpuBenchResult run_gpu_bench(double duration_s, int num_buffers, bool with_output) {
  FakeProcessorState state;
  DummyFinalPacketData<T> packet_data;

  BeamWeightsT<T> h_weights{};
  for (size_t ch = 0; ch < T::NR_CHANNELS; ++ch)
    for (size_t rx = 0; rx < T::NR_RECEIVERS; ++rx)
      for (size_t pol = 0; pol < T::NR_POLARIZATIONS; ++pol)
        for (size_t bm = 0; bm < T::NR_BEAMS; ++bm)
          h_weights.weights[ch][pol][bm][rx] =
              std::complex<__half>(__float2half(1.0f), __float2half(0.0f));

  BeamSteering<T> beam_steering({}, {}, {}, FrequencyPlan{}, 0, ArrayLocation{},
                                 180.0, 5);
  LambdaCorrBeamOnlyGPUPipeline<T> pipeline(num_buffers, &h_weights,
                                             std::move(beam_steering));
  pipeline.set_state(&state);

  std::shared_ptr<SingleHostMemoryOutput<T>> output;
  if (with_output) {
    output = std::make_shared<SingleHostMemoryOutput<T>>();
    pipeline.set_output(output);
  }
  // output_ == nullptr → no D2H memcpy for beam data; pure GPU compute throughput

  unsigned long long pipeline_runs = 0;
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() <
         duration_s) {
    pipeline.execute_pipeline(&packet_data);
    ++pipeline_runs;
  }
  cudaDeviceSynchronize();
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  constexpr size_t in_bytes  = sizeof(typename T::InputPacketSamplesType);
  constexpr size_t out_bytes = sizeof(typename T::BeamOutputType);
  const size_t effective_out = with_output ? out_bytes : 0;

  GpuBenchResult r{};
  r.nr_channels     = T::NR_CHANNELS;
  r.nr_fpga_sources = T::NR_FPGA_SOURCES;
  r.nr_receivers    = T::NR_RECEIVERS;
  r.runs            = pipeline_runs;
  r.elapsed         = elapsed;
  r.runs_per_sec    = pipeline_runs / elapsed;
  r.input_bytes     = in_bytes;
  r.output_bytes    = effective_out;
  r.input_gb_per_sec  = static_cast<double>(in_bytes)  * pipeline_runs / elapsed / 1e9;
  r.output_gb_per_sec = static_cast<double>(effective_out) * pipeline_runs / elapsed / 1e9;
  r.total_gb_per_sec  = static_cast<double>(in_bytes + effective_out) * pipeline_runs / elapsed / 1e9;
  return r;
}

static void print_result(const GpuBenchResult &r) {
  std::printf(
      "[CorrBeam ch=%zu fpga=%zu rx=%zu] "
      "elapsed=%.3f runs=%llu runs/sec=%.4f "
      "input_bytes=%zu output_bytes=%zu "
      "input_GB/sec=%.6f output_GB/sec=%.6f GB/sec=%.6f\n",
      r.nr_channels, r.nr_fpga_sources, r.nr_receivers,
      r.elapsed, (unsigned long long)r.runs, r.runs_per_sec,
      r.input_bytes, r.output_bytes,
      r.input_gb_per_sec, r.output_gb_per_sec, r.total_gb_per_sec);
  std::fflush(stdout);
}

// --------------------------------------------------------------------------
// LambdaGPUPipeline benchmark (full pipeline: ingest → permute → correlate
// → eigendecompose → beamform → FFT → downsample)
// --------------------------------------------------------------------------

struct LambdaGpuBenchResult {
  size_t nr_channels;
  size_t nr_fpga_sources;
  size_t nr_receivers;
  unsigned long long runs;
  double elapsed;
  double runs_per_sec;
  size_t input_bytes;
  double input_gb_per_sec;
  double avg_gpu_ms;
  double gpu_util;
};

template <typename T>
LambdaGpuBenchResult run_lambda_bench(double duration_s, int num_buffers) {
  FakeProcessorState state;
  DummyFinalPacketData<T> packet_data;

  BeamWeightsT<T> h_weights{};
  for (size_t ch = 0; ch < T::NR_CHANNELS; ++ch)
    for (size_t rx = 0; rx < T::NR_RECEIVERS; ++rx)
      for (size_t pol = 0; pol < T::NR_POLARIZATIONS; ++pol)
        for (size_t bm = 0; bm < T::NR_BEAMS; ++bm)
          h_weights.weights[ch][pol][bm][rx] =
              std::complex<__half>(__float2half(1.0f), __float2half(0.0f));

  BeamSteering<T> beam_steering({}, {}, {}, FrequencyPlan{}, 0, ArrayLocation{},
                                 180.0, 5);
  LambdaGPUPipeline<T> pipeline(num_buffers, &h_weights, std::move(beam_steering));
  pipeline.set_state(&state);

  unsigned long long pipeline_runs = 0;
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() <
         duration_s) {
    pipeline.execute_pipeline(&packet_data);
    ++pipeline_runs;
  }
  cudaDeviceSynchronize();
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  // avg_gpu_ms from the last NR_BENCHMARKING_RUNS events (the ring wraps modulo that).
  constexpr unsigned long long RING = LambdaGPUPipeline<T>::NR_BENCHMARKING_RUNS;
  const unsigned long long sample_runs = std::min(pipeline_runs, RING);
  double total_gpu_ms = 0.0;
  for (unsigned long long i = 0; i < sample_runs; ++i) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, pipeline.start_run[i], pipeline.stop_run[i]);
    total_gpu_ms += ms;
  }
  const double avg_gpu_ms = sample_runs > 0 ? total_gpu_ms / sample_runs : 0.0;

  constexpr size_t in_bytes = sizeof(typename T::InputPacketSamplesType);

  LambdaGpuBenchResult r{};
  r.nr_channels     = T::NR_CHANNELS;
  r.nr_fpga_sources = T::NR_FPGA_SOURCES;
  r.nr_receivers    = T::NR_RECEIVERS;
  r.runs            = pipeline_runs;
  r.elapsed         = elapsed;
  r.runs_per_sec    = pipeline_runs / elapsed;
  r.input_bytes     = in_bytes;
  r.input_gb_per_sec = static_cast<double>(in_bytes) * pipeline_runs / elapsed / 1e9;
  r.avg_gpu_ms      = avg_gpu_ms;
  r.gpu_util        = avg_gpu_ms * r.runs_per_sec / 1000.0;
  return r;
}

// Phase 0 relocation-vs-real-pipeline check (see
// /home/ubuntu/.claude/plans/i-want-to-start-breezy-lampson.md): the
// bench_gpudirect_scatter binary showed a large throughput deficit under a
// *synthetic* SM-saturating busy kernel. This re-runs the same relocation
// streams concurrently with the actual LambdaGPUPipeline<T> (the real
// correlate+beamform+eigen+fft workload) instead of a synthetic proxy, to
// check whether that finding holds against genuine pipeline occupancy
// patterns (which may leave more/less real scheduling gaps than a blunt
// fixed-grid FMA loop does).
template <typename T>
void run_relocation_check(double duration_s, int num_buffers, int num_streams,
                          int poll_batch, int dest_slots, double channels_pkt_rate) {
  FakeProcessorState state;
  DummyFinalPacketData<T> packet_data;

  BeamWeightsT<T> h_weights{};
  for (size_t ch = 0; ch < T::NR_CHANNELS; ++ch)
    for (size_t rx = 0; rx < T::NR_RECEIVERS; ++rx)
      for (size_t pol = 0; pol < T::NR_POLARIZATIONS; ++pol)
        for (size_t bm = 0; bm < T::NR_BEAMS; ++bm)
          h_weights.weights[ch][pol][bm][rx] =
              std::complex<__half>(__float2half(1.0f), __float2half(0.0f));

  BeamSteering<T> beam_steering({}, {}, {}, FrequencyPlan{}, 0, ArrayLocation{},
                                 180.0, 5);
  LambdaGPUPipeline<T> pipeline(num_buffers, &h_weights, std::move(beam_steering));
  pipeline.set_state(&state);

  auto run_pipeline_for = [&](double secs, std::atomic<bool> *stop_flag) {
    unsigned long long runs = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
      if (stop_flag && stop_flag->load(std::memory_order_relaxed)) break;
      pipeline.execute_pipeline(&packet_data);
      ++runs;
    }
    cudaDeviceSynchronize();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return runs / elapsed;
  };

  std::cout << "\n=== Relocation check [ch=" << T::NR_CHANNELS << " fpga=" << T::NR_FPGA_SOURCES
            << "]: baseline LambdaGPUPipeline alone (" << duration_s << "s) ===\n";
  const double baseline_runs_per_sec = run_pipeline_for(duration_s, nullptr);
  std::cout << "baseline pipeline runs/sec=" << baseline_runs_per_sec << "\n";

  const double target_per_stream = channels_pkt_rate / poll_batch;
  const double target_aggregate = target_per_stream * num_streams;
  const double pass_threshold = 2.0 * target_aggregate;
  std::cout << "\n=== Concurrent: real LambdaGPUPipeline + " << num_streams
            << " relocation streams (" << duration_s << "s) ===\n"
            << "  target ~" << target_per_stream << " replays/sec/stream, ~" << target_aggregate
            << " aggregate, pass threshold (2x)=" << pass_threshold << "\n";

  std::atomic<bool> start_gate{false};
  std::atomic<bool> stop_pipeline{false};
  std::vector<gpudirect_bench::StreamStats> stats(num_streams);
  std::vector<std::thread> threads;
  for (int i = 0; i < num_streams; ++i) {
    threads.emplace_back(gpudirect_bench::run_stream, poll_batch, dest_slots, duration_s,
                         std::ref(start_gate), std::ref(stats[i]));
  }
  double concurrent_runs_per_sec = 0.0;
  std::thread pipeline_thread([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    concurrent_runs_per_sec = run_pipeline_for(duration_s, &stop_pipeline);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  start_gate.store(true, std::memory_order_release);
  for (auto &t : threads) t.join();
  stop_pipeline.store(true, std::memory_order_release);
  pipeline_thread.join();
  std::cout << "concurrent pipeline runs/sec=" << concurrent_runs_per_sec << "\n";

  unsigned long long total_replays = 0, total_stalls = 0;
  for (int i = 0; i < num_streams; ++i) {
    const auto &s = stats[i];
    std::cout << "[stream " << i << "] replays/sec=" << (s.replays / duration_s)
              << " stalls=" << s.stalls << " avg_us=" << (s.replays ? s.total_replay_us / s.replays : 0.0)
              << " max_us=" << s.max_replay_us << "\n";
    total_replays += s.replays;
    total_stalls += s.stalls;
  }
  const double aggregate_rate = total_replays / duration_s;
  const double degradation_pct =
      baseline_runs_per_sec > 0 ? 100.0 * (1.0 - concurrent_runs_per_sec / baseline_runs_per_sec) : 0.0;

  std::cout << "\n[aggregate] replocation replays/sec=" << aggregate_rate
            << " total_stalls=" << total_stalls
            << " pipeline degradation=" << degradation_pct << "%\n";
  std::cout << "\n=== Phase 0 pass criteria (real pipeline) ===\n";
  bool pass = true;
  if (aggregate_rate >= pass_threshold) {
    std::cout << "[PASS] aggregate replay rate " << aggregate_rate << " >= 2x target (" << pass_threshold << ")\n";
  } else {
    std::cout << "[FAIL] aggregate replay rate " << aggregate_rate << " < 2x target (" << pass_threshold << ")\n";
    pass = false;
  }
  if (total_stalls == 0) {
    std::cout << "[PASS] zero double-buffer stalls\n";
  } else {
    std::cout << "[FAIL] " << total_stalls << " double-buffer stalls observed\n";
    pass = false;
  }
  if (degradation_pct < 10.0) {
    std::cout << "[PASS] pipeline degradation " << degradation_pct << "% < 10%\n";
  } else {
    std::cout << "[FAIL] pipeline degradation " << degradation_pct << "% >= 10%\n";
    pass = false;
  }
  std::cout << "\nOverall Phase 0 (real pipeline): " << (pass ? "PASS" : "FAIL") << "\n";
}

static void print_lambda_result(const LambdaGpuBenchResult &r) {
  std::printf(
      "[LambdaGPU ch=%zu fpga=%zu rx=%zu] "
      "elapsed=%.3f runs=%llu runs/sec=%.4f "
      "input_bytes=%zu input_GB/sec=%.6f "
      "avg_gpu_ms=%.3f gpu_util=%.1f%%\n",
      r.nr_channels, r.nr_fpga_sources, r.nr_receivers,
      r.elapsed, (unsigned long long)r.runs, r.runs_per_sec,
      r.input_bytes, r.input_gb_per_sec,
      r.avg_gpu_ms, r.gpu_util * 100.0);
  std::fflush(stdout);
}

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("bench_gpu");
  program.add_argument("--duration")
      .help("Duration in seconds to run the GPU pipeline per config")
      .default_value(30.0)
      .scan<'g', double>();
  program.add_argument("--num-buffers")
      .help("Deprecated; accepted for compatibility, 4-FPGA sweeps use --num-buffers-4fpga")
      .default_value(5)
      .scan<'i', int>();
  program.add_argument("--num-buffers-4fpga")
      .help("Pipeline double-buffer slots for 4-FPGA (large) configs; "
            "default 3 (sweet spot on 8 GB VRAM; 4+ buffers OOM at 32ch/4fpga)")
      .default_value(3)
      .scan<'i', int>();
  program.add_argument("--with-output")
      .help("Enable beam D2H output (measures GPU+PCIe; default: GPU-only throughput)")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--lambda-only")
      .help("Run only the LambdaGPU (full pipeline) sweep, skip CorrBeamOnly sweeps")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--corrbeam-only")
      .help("Run only the CorrBeamOnly sweeps, skip LambdaGPU sweep")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--relocation-check")
      .help("Phase 0 check: run GPUDirect relocation streams concurrently with the "
            "real LambdaGPUPipeline (Cfg32ch4fpga) instead of the sweeps above")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--relocation-streams")
      .help("Number of simulated FPGA relocation streams for --relocation-check")
      .default_value(4)
      .scan<'i', int>();
  program.add_argument("--relocation-poll-batch")
      .help("Packets per relocation batch for --relocation-check")
      .default_value(64)
      .scan<'i', int>();
  program.add_argument("--relocation-pkt-rate-per-channel")
      .help("Packets/sec/channel target for --relocation-check")
      .default_value(15500.0)
      .scan<'g', double>();
  program.add_argument("--fine-channel-check")
      .help("Compare LambdaGPU throughput with fine channelization off vs. on "
            "(NR_FINE_CHANNELS=32, edge-trim=2) at 8ch and 16ch/4fpga, instead "
            "of the sweeps above")
      .default_value(false)
      .implicit_value(true);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << "\n" << program;
    return 1;
  }

  const double duration_s        = program.get<double>("--duration");
  (void)program.get<int>("--num-buffers"); // accepted for CLI compatibility
  const int    num_buffers_4fpga = program.get<int>("--num-buffers-4fpga");
  const bool   with_output       = program.get<bool>("--with-output");
  const bool   lambda_only       = program.get<bool>("--lambda-only");
  const bool   corrbeam_only     = program.get<bool>("--corrbeam-only");
  const bool   relocation_check  = program.get<bool>("--relocation-check");
  const bool   fine_channel_check = program.get<bool>("--fine-channel-check");
  const int    relocation_streams = program.get<int>("--relocation-streams");
  const int    relocation_poll_batch = program.get<int>("--relocation-poll-batch");
  const double relocation_pkt_rate = program.get<double>("--relocation-pkt-rate-per-channel");

  // Async file logger so the pipeline's INFO_LOG/DEBUG_LOG macros have
  // somewhere to write without polluting stdout.
  static auto tp = std::make_shared<spdlog::details::thread_pool>(8192, 1);
  auto logger = std::make_shared<spdlog::async_logger>(
      "bench_gpu",
      std::make_shared<spdlog::sinks::basic_file_sink_mt>("app.log", false),
      tp, spdlog::async_overflow_policy::overrun_oldest);
  logger->set_level(spdlog::level::info);
  spatial::Logger::set(logger);

  if (relocation_check) {
    const double channels_pkt_rate = Cfg32ch4fpga::NR_CHANNELS * relocation_pkt_rate;
    run_relocation_check<Cfg32ch4fpga>(duration_s, num_buffers_4fpga, relocation_streams,
                                       relocation_poll_batch, /*dest_slots=*/256, channels_pkt_rate);
    return 0;
  }

  if (fine_channel_check) {
    std::cout << "=== Fine-channelization effect on LambdaGPUPipeline throughput ===\n"
              << "  duration=" << duration_s << "s each, num_buffers=" << num_buffers_4fpga << "\n\n";

    auto compare = [&](const char *label, auto off_result, auto on_result) {
      std::cout << "--- " << label << " ---\n";
      std::cout << "  off (NR_FINE_CHANNELS=1):  "; print_lambda_result(off_result);
      std::cout << "  on  (NR_FINE_CHANNELS=32, effective ch=" << on_result.nr_channels << "): ";
      print_lambda_result(on_result);
      const double throughput_ratio = on_result.runs_per_sec / off_result.runs_per_sec;
      const double gpu_ms_ratio = on_result.avg_gpu_ms / off_result.avg_gpu_ms;
      std::cout << "  => runs/sec ratio (on/off) = " << throughput_ratio
                << "  (" << (1.0 / throughput_ratio) << "x slower)"
                << " | avg_gpu_ms ratio (on/off) = " << gpu_ms_ratio << "x\n\n";
    };

    std::cout << "[8 coarse channels]\n";
    compare("8ch4fpga", run_lambda_bench<Cfg8ch4fpga>(duration_s, num_buffers_4fpga),
           run_lambda_bench<Cfg8ch4fpga_fine32>(duration_s, num_buffers_4fpga));

    std::cout << "[16 coarse channels]\n";
    compare("16ch4fpga", run_lambda_bench<Cfg16ch4fpga>(duration_s, num_buffers_4fpga),
           run_lambda_bench<Cfg16ch4fpga_fine32>(duration_s, num_buffers_4fpga));

    return 0;
  }

  std::cout << "bench_gpu: 4-FPGA CorrBeam channel/corr-packet sweeps + 4-FPGA LambdaGPU channel sweep"
            << "\n  duration=" << duration_s << "s each"
            << "  num_buffers(4fpga)=" << num_buffers_4fpga
            << "  with_output=" << (with_output ? "yes" : "no")
            << "\nNOTE: first run per config triggers TCC NVRTC JIT compilation "
               "(cached for subsequent runs)\n";

  if (!lambda_only) {
    std::cout << "\n=== CorrBeamOnly: 4-FPGA channel sweep (NR_PACKETS_FOR_CORRELATION=256) ===\n";
    print_result(run_gpu_bench<Cfg8ch4fpga> (duration_s, num_buffers_4fpga, with_output));
    print_result(run_gpu_bench<Cfg16ch4fpga>(duration_s, num_buffers_4fpga, with_output));
    print_result(run_gpu_bench<Cfg24ch4fpga>(duration_s, num_buffers_4fpga, with_output));
    print_result(run_gpu_bench<Cfg32ch4fpga>(duration_s, num_buffers_4fpga, with_output));

    std::cout << "\n=== CorrBeamOnly: 4-FPGA corr-packet sweep (8ch, corr=64..1024) ===\n";
    print_result(run_gpu_bench<Cfg8ch4fpga_c64>  (duration_s, num_buffers_4fpga, with_output));
    print_result(run_gpu_bench<Cfg8ch4fpga_c128> (duration_s, num_buffers_4fpga, with_output));
    print_result(run_gpu_bench<Cfg8ch4fpga>      (duration_s, num_buffers_4fpga, with_output));
    print_result(run_gpu_bench<Cfg8ch4fpga_c512> (duration_s, num_buffers_4fpga, with_output));
    print_result(run_gpu_bench<Cfg8ch4fpga_c1024>(duration_s, num_buffers_4fpga, with_output));
  }

  if (!corrbeam_only) {
    std::cout << "\n=== LambdaGPU (full: corr+beam+eigen+fft): 4-FPGA channel sweep ===\n";
    print_lambda_result(run_lambda_bench<Cfg8ch4fpga> (duration_s, num_buffers_4fpga));
    print_lambda_result(run_lambda_bench<Cfg16ch4fpga>(duration_s, num_buffers_4fpga));
    print_lambda_result(run_lambda_bench<Cfg24ch4fpga>(duration_s, num_buffers_4fpga));
    print_lambda_result(run_lambda_bench<Cfg32ch4fpga>(duration_s, num_buffers_4fpga));
  }

  return 0;
}
