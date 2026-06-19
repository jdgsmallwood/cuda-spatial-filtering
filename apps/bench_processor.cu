#include <argparse/argparse.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstdio>
#include <iostream>
#include <memory>
#include <thread>
#include <unordered_map>

#include "spatial/logging.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.hpp"
#include "synthetic_packets.hpp"
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>

// Fixed representative configs for the LAMBDA instrument.  These are
// compile-time constants so the same binary benchmarks all shapes without
// needing to rebuild with different -DNR_OBSERVING_* values.
//                                    ch  fp   ts   rx  pol rxpp corr  bm  pad  blk       acc
using Cfg1ch1fpga  = LambdaConfig<    1,  1,   64,  10,  2,  10, 256,  1,  32,  32, 10000000>;
using Cfg8ch1fpga  = LambdaConfig<    8,  1,   64,  10,  2,  10, 256,  1,  32,  32, 10000000>;
using Cfg16ch1fpga = LambdaConfig<   16,  1,   64,  10,  2,  10, 256,  1,  32,  32, 10000000>;
using Cfg24ch1fpga = LambdaConfig<   24,  1,   64,  10,  2,  10, 256,  1,  32,  32, 10000000>;
using Cfg32ch1fpga = LambdaConfig<   32,  1,   64,  10,  2,  10, 256,  1,  32,  32, 10000000>;
using Cfg8ch4fpga  = LambdaConfig<    8,  4,   64,  40,  2,  10, 256,  1,  64,  32, 10000000>;
using Cfg16ch4fpga = LambdaConfig<   16,  4,   64,  40,  2,  10, 256,  1,  64,  32, 10000000>;
using Cfg32ch4fpga = LambdaConfig<   32,  4,   64,  40,  2,  10, 256,  1,  64,  32, 10000000>;

struct BenchResult {
  size_t nr_channels;
  size_t nr_fpga_sources;
  size_t nr_receivers;
  size_t bytes_per_packet;
  uint64_t packets_fed;
  uint64_t packets_processed;
  uint64_t packets_missing;
  uint64_t packets_discarded;
  uint64_t buffers_completed;
  double elapsed;
  double packets_per_sec;
  double gb_per_sec;
};

// Stands in for the GPU pipeline: immediately hands the buffer straight back
// to the processor, so this benchmark measures only the
// capture-ring -> reassembly path (ProcessorState::process_packets +
// pipeline_feeder), not GPU execution time.
class NullPipeline : public GPUPipeline {
public:
  std::atomic<uint64_t> buffers_completed{0};

  void execute_pipeline(FinalPacketData *packet_data,
                        const bool = false) override {
    buffers_completed.fetch_add(1, std::memory_order_relaxed);
    state_->release_buffer(static_cast<int>(packet_data->buffer_index));
  }

  void dump_visibilities(const uint64_t = 0) override {}
};

template <typename Config>
BenchResult run_processor_bench(double duration_s) {
  constexpr size_t PKT_BYTES =
      sizeof(EthernetHeader) + sizeof(IPHeader) + sizeof(UDPHeader) +
      sizeof(CustomHeader) + sizeof(typename Config::PacketPayloadType);

  std::array<int64_t, Config::NR_FPGA_SOURCES> fpga_delays{};
  std::unordered_map<uint32_t, int> fpga_ids;
  for (size_t i = 0; i < Config::NR_FPGA_SOURCES; ++i) {
    fpga_ids[static_cast<uint32_t>(i)] = static_cast<int>(i);
  }

  constexpr size_t NR_INPUT_BUFFERS = 8;
  constexpr size_t RING_BUFFER_SIZE = 200000;
  ProcessorState<Config, NR_INPUT_BUFFERS, RING_BUFFER_SIZE> state(
      Config::NR_PACKETS_FOR_CORRELATION, Config::NR_TIME_STEPS_PER_PACKET,
      /*min_freq_channel=*/0, fpga_delays, fpga_ids);

  NullPipeline pipeline;
  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);

  std::thread processor([&state]() { state.process_packets(); });
  std::thread feeder([&state]() { state.pipeline_feeder(); });

  const auto sample_fn = [](int, int, int) {
    return std::complex<int8_t>(2, -2);
  };
  const auto scale_fn = [](int, int) { return static_cast<int16_t>(1); };

  constexpr uint64_t NR_BETWEEN_SAMPLES = Config::NR_TIME_STEPS_PER_PACKET;
  const uint64_t start_sample = NR_BETWEEN_SAMPLES * 10;

  uint64_t round = 0;
  uint64_t packets_fed = 0;
  const auto start = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start)
             .count() < duration_s) {
    const uint64_t round_start_sample =
        start_sample +
        round * Config::NR_PACKETS_FOR_CORRELATION * NR_BETWEEN_SAMPLES;

    for (size_t channel = 0; channel < Config::NR_CHANNELS; ++channel) {
      for (size_t fpga = 0; fpga < Config::NR_FPGA_SOURCES; ++fpga) {
        for (int pkt = 0;
             pkt <= static_cast<int>(Config::NR_PACKETS_FOR_CORRELATION);
             ++pkt) {
          const uint64_t sample_count =
              round_start_sample +
              static_cast<uint64_t>(pkt) * NR_BETWEEN_SAMPLES;
          test_support::feed_lambda_packet<Config>(
              state, sample_count, static_cast<uint32_t>(fpga),
              static_cast<uint16_t>(channel), sample_fn, scale_fn);
          packets_fed++;
        }
        // The "-1" packet (one before the buffer start) is fed last, mirroring
        // SyntheticPipelineRun's proven fill order.
        const uint64_t sample_count_m1 =
            round_start_sample - NR_BETWEEN_SAMPLES;
        test_support::feed_lambda_packet<Config>(
            state, sample_count_m1, static_cast<uint32_t>(fpga),
            static_cast<uint16_t>(channel), sample_fn, scale_fn);
        packets_fed++;
      }
    }
    round++;
  }

  state.shutdown();
  processor.join();
  feeder.join();

  const auto end = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration<double>(end - start).count();
  const uint64_t packets_processed = state.packets_processed.load();

  BenchResult r;
  r.nr_channels = Config::NR_CHANNELS;
  r.nr_fpga_sources = Config::NR_FPGA_SOURCES;
  r.nr_receivers = Config::NR_RECEIVERS;
  r.bytes_per_packet = PKT_BYTES;
  r.packets_fed = packets_fed;
  r.packets_processed = packets_processed;
  r.packets_missing = state.packets_missing;
  r.packets_discarded = state.packets_discarded.load();
  r.buffers_completed = pipeline.buffers_completed.load();
  r.elapsed = elapsed;
  r.packets_per_sec = packets_processed / elapsed;
  r.gb_per_sec =
      static_cast<double>(packets_processed) * PKT_BYTES / elapsed / 1e9;
  return r;
}

static void print_result(const BenchResult &r) {
  std::printf(
      "[Processor ch=%zu fpga=%zu rx=%zu] "
      "packets_fed=%llu packets_processed=%llu packets_missing=%llu "
      "packets_discarded=%llu buffers_completed=%llu "
      "elapsed=%.3f bytes_per_packet=%zu "
      "packets/sec=%.2f GB/sec=%.6f\n",
      r.nr_channels, r.nr_fpga_sources, r.nr_receivers,
      (unsigned long long)r.packets_fed,
      (unsigned long long)r.packets_processed,
      (unsigned long long)r.packets_missing,
      (unsigned long long)r.packets_discarded,
      (unsigned long long)r.buffers_completed, r.elapsed, r.bytes_per_packet,
      r.packets_per_sec, r.gb_per_sec);
  std::fflush(stdout);
}

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("bench_processor");
  program.add_argument("--duration")
      .help("Duration in seconds to feed synthetic packets per config")
      .default_value(10.0)
      .scan<'g', double>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  const double duration_s = program.get<double>("--duration");

  // Redirect the default stdout spdlog logger to a file so INFO_LOG calls
  // from ProcessorState don't pollute the benchmark's stdout output.
  static auto tp = std::make_shared<spdlog::details::thread_pool>(8192, 1);
  auto logger = std::make_shared<spdlog::async_logger>(
      "bench_processor",
      std::make_shared<spdlog::sinks::basic_file_sink_mt>("app.log", false),
      tp, spdlog::async_overflow_policy::overrun_oldest);
  logger->set_level(spdlog::level::warn);
  spatial::Logger::set(logger);

  std::cout << "bench_processor: 8 configs x " << duration_s << "s each"
            << std::endl;

  print_result(run_processor_bench<Cfg1ch1fpga> (duration_s));
  print_result(run_processor_bench<Cfg8ch1fpga> (duration_s));
  print_result(run_processor_bench<Cfg16ch1fpga>(duration_s));
  print_result(run_processor_bench<Cfg24ch1fpga>(duration_s));
  print_result(run_processor_bench<Cfg32ch1fpga>(duration_s));
  print_result(run_processor_bench<Cfg8ch4fpga> (duration_s));
  print_result(run_processor_bench<Cfg16ch4fpga>(duration_s));
  print_result(run_processor_bench<Cfg32ch4fpga>(duration_s));

  return 0;
}
