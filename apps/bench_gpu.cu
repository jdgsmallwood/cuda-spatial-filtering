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

// Fixed representative GPU pipeline configs -- 8..32 channels in steps of 8
// for both 1-FPGA (10 rx) and 4-FPGA (40 rx) layouts.  Each config is a
// separate template instantiation so a single binary benchmarks all shapes
// without needing different -DNR_OBSERVING_* CMake builds.
//
// NR_PADDED_RECEIVERS must be the next multiple of 32 ≥ NR_RECEIVERS:
//   10 rx  → 32 padded   (1 fpga)
//   40 rx  → 64 padded   (4 fpga)
//
//                                     ch  fp   ts   rx  pol rxpp corr  bm  pad  blk       acc
using Cfg8ch1fpga  = LambdaConfig<     8,  1,   64,  10,  2,  10, 256,  1,  32,  32, 10000000>;
using Cfg16ch1fpga = LambdaConfig<    16,  1,   64,  10,  2,  10, 256,  1,  32,  32, 10000000>;
using Cfg24ch1fpga = LambdaConfig<    24,  1,   64,  10,  2,  10, 256,  1,  32,  32, 10000000>;
using Cfg32ch1fpga = LambdaConfig<    32,  1,   64,  10,  2,  10, 256,  1,  32,  32, 10000000>;
using Cfg8ch4fpga  = LambdaConfig<     8,  4,   64,  40,  2,  10, 256,  1,  64,  32, 10000000>;
using Cfg16ch4fpga = LambdaConfig<    16,  4,   64,  40,  2,  10, 256,  1,  64,  32, 10000000>;
using Cfg24ch4fpga = LambdaConfig<    24,  4,   64,  40,  2,  10, 256,  1,  64,  32, 10000000>;
using Cfg32ch4fpga = LambdaConfig<    32,  4,   64,  40,  2,  10, 256,  1,  64,  32, 10000000>;

// Correlation-packet sweep: 8ch/1fpga and 8ch/4fpga held fixed while
// NR_PACKETS_FOR_CORRELATION varies 64→1024 (powers of 2).  The 256-packet
// case reuses the configs above.
//                                       ch  fp   ts   rx  pol rxpp  corr   bm  pad  blk       acc
using Cfg8ch1fpga_c64   = LambdaConfig<  8,  1,   64,  10,  2,  10,   64,  1,  32,  32, 10000000>;
using Cfg8ch1fpga_c128  = LambdaConfig<  8,  1,   64,  10,  2,  10,  128,  1,  32,  32, 10000000>;
using Cfg8ch1fpga_c512  = LambdaConfig<  8,  1,   64,  10,  2,  10,  512,  1,  32,  32, 10000000>;
using Cfg8ch1fpga_c1024 = LambdaConfig<  8,  1,   64,  10,  2,  10, 1024,  1,  32,  32, 10000000>;
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

  // Run exactly NR_BENCHMARKING_RUNS times (or until duration_s), so the
  // event ring doesn't wrap and all start/stop pairs remain queryable.
  constexpr unsigned long long MAX_RUNS = LambdaGPUPipeline<T>::NR_BENCHMARKING_RUNS;
  unsigned long long pipeline_runs = 0;
  const auto t0 = std::chrono::steady_clock::now();
  while (pipeline_runs < MAX_RUNS &&
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() <
             duration_s) {
    pipeline.execute_pipeline(&packet_data);
    ++pipeline_runs;
  }
  cudaDeviceSynchronize();
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  double total_gpu_ms = 0.0;
  for (unsigned long long i = 0; i < pipeline_runs; ++i) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, pipeline.start_run[i], pipeline.stop_run[i]);
    total_gpu_ms += ms;
  }
  const double avg_gpu_ms = pipeline_runs > 0 ? total_gpu_ms / pipeline_runs : 0.0;

  LambdaGpuBenchResult r{};
  r.nr_channels     = T::NR_CHANNELS;
  r.nr_fpga_sources = T::NR_FPGA_SOURCES;
  r.nr_receivers    = T::NR_RECEIVERS;
  r.runs            = pipeline_runs;
  r.elapsed         = elapsed;
  r.runs_per_sec    = pipeline_runs / elapsed;
  r.avg_gpu_ms      = avg_gpu_ms;
  r.gpu_util        = avg_gpu_ms * r.runs_per_sec / 1000.0;
  return r;
}

static void print_lambda_result(const LambdaGpuBenchResult &r) {
  std::printf(
      "[LambdaGPU ch=%zu fpga=%zu rx=%zu] "
      "elapsed=%.3f runs=%llu runs/sec=%.4f "
      "avg_gpu_ms=%.3f gpu_util=%.1f%%\n",
      r.nr_channels, r.nr_fpga_sources, r.nr_receivers,
      r.elapsed, (unsigned long long)r.runs, r.runs_per_sec,
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
      .help("Pipeline double-buffer slots for 1-FPGA (small) configs")
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

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << "\n" << program;
    return 1;
  }

  const double duration_s       = program.get<double>("--duration");
  const int    num_buffers      = program.get<int>("--num-buffers");
  const int    num_buffers_4fpga = program.get<int>("--num-buffers-4fpga");
  const bool   with_output      = program.get<bool>("--with-output");

  // Async file logger so the pipeline's INFO_LOG/DEBUG_LOG macros have
  // somewhere to write without polluting stdout.
  static auto tp = std::make_shared<spdlog::details::thread_pool>(8192, 1);
  auto logger = std::make_shared<spdlog::async_logger>(
      "bench_gpu",
      std::make_shared<spdlog::sinks::basic_file_sink_mt>("app.log", false),
      tp, spdlog::async_overflow_policy::overrun_oldest);
  logger->set_level(spdlog::level::info);
  spatial::Logger::set(logger);

  std::cout << "bench_gpu: CorrBeam channel/corr-packet sweeps + LambdaGPU channel sweep"
            << "\n  duration=" << duration_s << "s each"
            << "  num_buffers(1fpga)=" << num_buffers
            << "  num_buffers(4fpga)=" << num_buffers_4fpga
            << "  with_output=" << (with_output ? "yes" : "no")
            << "\nNOTE: first run per config triggers TCC NVRTC JIT compilation "
               "(cached for subsequent runs)\n";

  std::cout << "\n=== CorrBeamOnly: channel sweep (NR_PACKETS_FOR_CORRELATION=256) ===\n";
  print_result(run_gpu_bench<Cfg8ch1fpga> (duration_s, num_buffers,       with_output));
  print_result(run_gpu_bench<Cfg16ch1fpga>(duration_s, num_buffers,       with_output));
  print_result(run_gpu_bench<Cfg24ch1fpga>(duration_s, num_buffers,       with_output));
  print_result(run_gpu_bench<Cfg32ch1fpga>(duration_s, num_buffers,       with_output));
  print_result(run_gpu_bench<Cfg8ch4fpga> (duration_s, num_buffers_4fpga, with_output));
  print_result(run_gpu_bench<Cfg16ch4fpga>(duration_s, num_buffers_4fpga, with_output));
  print_result(run_gpu_bench<Cfg24ch4fpga>(duration_s, num_buffers_4fpga, with_output));
  print_result(run_gpu_bench<Cfg32ch4fpga>(duration_s, num_buffers_4fpga, with_output));

  std::cout << "\n=== CorrBeamOnly: corr-packet sweep (8ch, corr=64..1024) ===\n";
  print_result(run_gpu_bench<Cfg8ch1fpga_c64>  (duration_s, num_buffers,       with_output));
  print_result(run_gpu_bench<Cfg8ch1fpga_c128> (duration_s, num_buffers,       with_output));
  print_result(run_gpu_bench<Cfg8ch1fpga>      (duration_s, num_buffers,       with_output));
  print_result(run_gpu_bench<Cfg8ch1fpga_c512> (duration_s, num_buffers,       with_output));
  print_result(run_gpu_bench<Cfg8ch1fpga_c1024>(duration_s, num_buffers,       with_output));
  print_result(run_gpu_bench<Cfg8ch4fpga_c64>  (duration_s, num_buffers_4fpga, with_output));
  print_result(run_gpu_bench<Cfg8ch4fpga_c128> (duration_s, num_buffers_4fpga, with_output));
  print_result(run_gpu_bench<Cfg8ch4fpga>      (duration_s, num_buffers_4fpga, with_output));
  print_result(run_gpu_bench<Cfg8ch4fpga_c512> (duration_s, num_buffers_4fpga, with_output));
  print_result(run_gpu_bench<Cfg8ch4fpga_c1024>(duration_s, num_buffers_4fpga, with_output));

  std::cout << "\n=== LambdaGPU (full: corr+beam+eigen+fft): channel sweep ===\n";
  print_lambda_result(run_lambda_bench<Cfg8ch1fpga> (duration_s, num_buffers));
  print_lambda_result(run_lambda_bench<Cfg16ch1fpga>(duration_s, num_buffers));
  print_lambda_result(run_lambda_bench<Cfg24ch1fpga>(duration_s, num_buffers));
  print_lambda_result(run_lambda_bench<Cfg32ch1fpga>(duration_s, num_buffers));
  print_lambda_result(run_lambda_bench<Cfg8ch4fpga> (duration_s, num_buffers_4fpga));
  print_lambda_result(run_lambda_bench<Cfg16ch4fpga>(duration_s, num_buffers_4fpga));
  print_lambda_result(run_lambda_bench<Cfg24ch4fpga>(duration_s, num_buffers_4fpga));
  print_lambda_result(run_lambda_bench<Cfg32ch4fpga>(duration_s, num_buffers_4fpga));

  return 0;
}
