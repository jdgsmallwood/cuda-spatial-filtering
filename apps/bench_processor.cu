#include <argparse/argparse.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "spatial/logging.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.hpp"
#include "synthetic_packets.hpp"
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>

#ifndef BENCH_PROCESSOR_RING_BUFFER_SIZE
#define BENCH_PROCESSOR_RING_BUFFER_SIZE 8192
#endif

#ifndef BENCH_PROCESSOR_WORKER_COUNT
#define BENCH_PROCESSOR_WORKER_COUNT 6
#endif

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

struct CopyBenchResult {
  const char *mode;
  int threads;
  uint64_t copies;
  size_t bytes_per_copy;
  double elapsed;
  double gb_per_sec;
};

template <typename Config>
CopyBenchResult run_copy_microbench(const char *mode, int thread_count,
                                    double duration_s) {
  constexpr size_t SAMPLE_BYTES = sizeof(typename Config::PacketDataStructure);
  constexpr size_t SLOTS_PER_THREAD = 2048;
  static_assert((SLOTS_PER_THREAD & (SLOTS_PER_THREAD - 1)) == 0);

  const bool use_nt = std::strcmp(mode, "nt") == 0;
  const size_t bytes_per_thread = SAMPLE_BYTES * SLOTS_PER_THREAD;

  std::vector<uint8_t *> src(thread_count, nullptr);
  std::vector<uint8_t *> dst(thread_count, nullptr);
  for (int t = 0; t < thread_count; ++t) {
    void *src_raw = nullptr;
    if (posix_memalign(&src_raw, 64, bytes_per_thread) != 0) {
      throw std::bad_alloc();
    }
    src[t] = static_cast<uint8_t *>(src_raw);
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&dst[t]),
                             bytes_per_thread, cudaHostAllocDefault));
    std::memset(src[t], 0x5a, bytes_per_thread);
    std::memset(dst[t], 0, bytes_per_thread);
  }

  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<uint64_t> copies(thread_count, 0);
  std::vector<std::thread> threads;
  threads.reserve(thread_count);

  for (int t = 0; t < thread_count; ++t) {
    threads.emplace_back([&, t]() {
      ready.fetch_add(1, std::memory_order_release);
      while (!go.load(std::memory_order_acquire)) {
        _mm_pause();
      }

      uint64_t local = 0;
      auto next_check = std::chrono::steady_clock::now();
      const auto deadline = next_check + std::chrono::duration<double>(duration_s);
      while (true) {
        for (int i = 0; i < 256; ++i) {
          const size_t slot = local & (SLOTS_PER_THREAD - 1);
          uint8_t *d = dst[t] + slot * SAMPLE_BYTES;
          const uint8_t *s = src[t] + slot * SAMPLE_BYTES;
          if (use_nt) {
            copy_nt_unfenced(d, s, SAMPLE_BYTES);
          } else {
            std::memcpy(d, s, SAMPLE_BYTES);
          }
          ++local;
        }
        next_check = std::chrono::steady_clock::now();
        if (next_check >= deadline) {
          break;
        }
      }
      if (use_nt) {
        _mm_sfence();
      }
      copies[t] = local;
    });
  }

  while (ready.load(std::memory_order_acquire) < thread_count) {
    _mm_pause();
  }
  const auto start = std::chrono::steady_clock::now();
  go.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }
  const auto end = std::chrono::steady_clock::now();

  uint64_t total_copies = 0;
  for (uint64_t c : copies) {
    total_copies += c;
  }
  for (int t = 0; t < thread_count; ++t) {
    cudaFreeHost(dst[t]);
    std::free(src[t]);
  }

  const double elapsed = std::chrono::duration<double>(end - start).count();
  return CopyBenchResult{
      .mode = mode,
      .threads = thread_count,
      .copies = total_copies,
      .bytes_per_copy = SAMPLE_BYTES,
      .elapsed = elapsed,
      .gb_per_sec =
          static_cast<double>(total_copies) * SAMPLE_BYTES / elapsed / 1e9};
}

static void print_copy_result(const CopyBenchResult &r) {
  std::printf("[CopyMicrobench mode=%s threads=%d] copies=%llu "
              "elapsed=%.3f bytes_per_copy=%zu GB/sec=%.6f\n",
              r.mode, r.threads, (unsigned long long)r.copies, r.elapsed,
              r.bytes_per_copy, r.gb_per_sec);
  std::fflush(stdout);
}

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
  // 8192 slots × ~2752 bytes = ~22 MB — fits within the machine's 32 MB L3,
  // dramatically reducing LLC miss rate.  The ring provides ~1.4× the
  // REGULAR_BATCH_SIZE (6000) of slack so the feeder very rarely stalls.
  constexpr size_t RING_BUFFER_SIZE = BENCH_PROCESSOR_RING_BUFFER_SIZE;
  constexpr int WORKER_COUNT = BENCH_PROCESSOR_WORKER_COUNT;
  ProcessorState<Config, NR_INPUT_BUFFERS, RING_BUFFER_SIZE, WORKER_COUNT> state(
      Config::NR_PACKETS_FOR_CORRELATION, Config::NR_TIME_STEPS_PER_PACKET,
      /*min_freq_channel=*/0, fpga_delays, fpga_ids);

  NullPipeline pipeline;
  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);

  // Synchronous mode: execute_pipeline() is called directly from
  // handle_buffer_completion() (not via the async SPSC queue).  This means
  // release_buffer() runs — and replenishes buffer_ordering_queue — BEFORE
  // advance_to_next_buffer() is called, so the condition-variable wait in
  // advance_to_next_buffer() never blocks.  Without this, pipeline_feeder()
  // exits on shutdown() before draining pipeline_q, leaving buffers unreleased
  // and advance_to_next_buffer() waiting on a cv that is never notified.
  state.synchronous_pipeline = true;

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

template <typename Config>
BenchResult run_direct_processor_bench(double duration_s) {
  constexpr size_t PKT_BYTES =
      sizeof(EthernetHeader) + sizeof(IPHeader) + sizeof(UDPHeader) +
      sizeof(CustomHeader) + sizeof(typename Config::PacketPayloadType);

  std::array<int64_t, Config::NR_FPGA_SOURCES> fpga_delays{};
  std::unordered_map<uint32_t, int> fpga_ids;
  for (size_t i = 0; i < Config::NR_FPGA_SOURCES; ++i) {
    fpga_ids[static_cast<uint32_t>(i)] = static_cast<int>(i);
  }

  constexpr size_t NR_INPUT_BUFFERS = 8;
  constexpr size_t RING_BUFFER_SIZE = BENCH_PROCESSOR_RING_BUFFER_SIZE;
  constexpr int WORKER_COUNT = BENCH_PROCESSOR_WORKER_COUNT;
  ProcessorState<Config, NR_INPUT_BUFFERS, RING_BUFFER_SIZE, WORKER_COUNT> state(
      Config::NR_PACKETS_FOR_CORRELATION, Config::NR_TIME_STEPS_PER_PACKET,
      /*min_freq_channel=*/0, fpga_delays, fpga_ids);

  NullPipeline pipeline;
  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);
  state.synchronous_pipeline = true;

  typename Config::PacketDataStructure sample_payload{};
  typename Config::PacketScaleStructure scale_payload{};
  for (size_t r = 0; r < Config::NR_RECEIVERS_PER_PACKET; ++r) {
    for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p) {
      scale_payload[r][p] = static_cast<int16_t>(1);
    }
  }
  for (size_t t = 0; t < Config::NR_TIME_STEPS_PER_PACKET; ++t) {
    for (size_t r = 0; r < Config::NR_RECEIVERS_PER_PACKET; ++r) {
      for (size_t p = 0; p < Config::NR_POLARIZATIONS; ++p) {
        sample_payload[t][r][p] = std::complex<int8_t>(2, -2);
      }
    }
  }

  constexpr uint64_t NR_BETWEEN_SAMPLES = Config::NR_TIME_STEPS_PER_PACKET;
  const uint64_t start_sample = NR_BETWEEN_SAMPLES * 10;

  state.initialize_for_direct_ingest(start_sample, 0);

  std::atomic<uint64_t> work_round{0};
  std::atomic<bool> stop{false};
  std::atomic<int> ready{0};
  std::atomic<int> next_buffer_hint{-1};
  std::array<std::atomic<uint64_t>, Config::NR_FPGA_SOURCES> completed_round;
  for (auto &completed : completed_round) {
    completed.store(0, std::memory_order_relaxed);
  }

  std::vector<std::thread> producers;
  producers.reserve(Config::NR_FPGA_SOURCES);
  for (size_t fpga = 0; fpga < Config::NR_FPGA_SOURCES; ++fpga) {
    producers.emplace_back([&, fpga]() {
      ready.fetch_add(1, std::memory_order_release);
      uint64_t local_round = 1;
      while (true) {
        while (work_round.load(std::memory_order_acquire) < local_round) {
          if (stop.load(std::memory_order_acquire)) {
            return;
          }
          _mm_pause();
        }
        if (stop.load(std::memory_order_acquire)) {
          return;
        }

        const uint64_t round_index = local_round - 1;
        const uint64_t round_start_sample =
            start_sample +
            round_index * Config::NR_PACKETS_FOR_CORRELATION *
                NR_BETWEEN_SAMPLES;
        uint64_t processed_local = 0;
        const int local_next_buffer_hint =
            next_buffer_hint.load(std::memory_order_acquire);
        for (size_t channel = 0; channel < Config::NR_CHANNELS; ++channel) {
          for (int pkt = 0;
               pkt <= static_cast<int>(Config::NR_PACKETS_FOR_CORRELATION);
               ++pkt) {
            const uint64_t sample_count =
                round_start_sample +
                static_cast<uint64_t>(pkt) * NR_BETWEEN_SAMPLES;
            processed_local += state.ingest_packet_direct(
                sample_count, static_cast<uint32_t>(fpga),
                static_cast<uint16_t>(channel), &sample_payload, &scale_payload,
                false, local_next_buffer_hint);
          }
          const uint64_t sample_count_m1 =
              round_start_sample - NR_BETWEEN_SAMPLES;
          processed_local += state.ingest_packet_direct(
              sample_count_m1, static_cast<uint32_t>(fpga),
              static_cast<uint16_t>(channel), &sample_payload, &scale_payload,
              false, local_next_buffer_hint);
        }
        if (processed_local > 0) {
          state.packets_processed.fetch_add(processed_local,
                                            std::memory_order_relaxed);
        }
        completed_round[fpga].store(local_round, std::memory_order_release);
        ++local_round;
      }
    });
  }

  while (ready.load(std::memory_order_acquire) <
         static_cast<int>(Config::NR_FPGA_SOURCES)) {
    _mm_pause();
  }

  uint64_t round = 0;
  uint64_t packets_fed = 0;
  constexpr uint64_t PACKETS_PER_ROUND =
      Config::NR_FPGA_SOURCES * Config::NR_CHANNELS *
      (Config::NR_PACKETS_FOR_CORRELATION + 2);
  const auto start = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start)
             .count() < duration_s) {
    const uint64_t next_round = round + 1;
    next_buffer_hint.store(state.direct_ingest_next_buffer_index(),
                           std::memory_order_release);
    work_round.store(next_round, std::memory_order_release);
    for (size_t fpga = 0; fpga < Config::NR_FPGA_SOURCES; ++fpga) {
      while (completed_round[fpga].load(std::memory_order_acquire) <
             next_round) {
        _mm_pause();
      }
    }
    packets_fed += PACKETS_PER_ROUND;
    state.handle_direct_ingest_ordered_buffer_complete();
    round = next_round;
  }

  stop.store(true, std::memory_order_release);
  work_round.fetch_add(1, std::memory_order_release);
  for (auto &producer : producers) {
    producer.join();
  }

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
  program.add_argument("--copy-microbench")
      .help("Run pinned-memory copy ceiling microbenchmark and exit")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--copy-threads")
      .help("Threads for --copy-microbench")
      .default_value(BENCH_PROCESSOR_WORKER_COUNT)
      .scan<'i', int>();
  program.add_argument("--config")
      .help("Processor config to run: all, 1x1, 8x1, 16x1, 24x1, 32x1, 8x4, 16x4, 32x4")
      .default_value(std::string("all"));
  program.add_argument("--mode")
      .help("Benchmark mode: ring or direct")
      .default_value(std::string("ring"));

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  const double duration_s = program.get<double>("--duration");
  const bool copy_microbench = program.get<bool>("--copy-microbench");
  const int copy_threads = program.get<int>("--copy-threads");
  const std::string config = program.get<std::string>("--config");
  const std::string mode = program.get<std::string>("--mode");

  // Redirect the default stdout spdlog logger to a file so INFO_LOG calls
  // from ProcessorState don't pollute the benchmark's stdout output.
  static auto tp = std::make_shared<spdlog::details::thread_pool>(8192, 1);
  auto logger = std::make_shared<spdlog::async_logger>(
      "bench_processor",
      std::make_shared<spdlog::sinks::basic_file_sink_mt>("app.log", false),
      tp, spdlog::async_overflow_policy::overrun_oldest);
  logger->set_level(spdlog::level::warn);
  spatial::Logger::set(logger);

  if (copy_microbench) {
    print_copy_result(
        run_copy_microbench<Cfg8ch4fpga>("memcpy", copy_threads, duration_s));
    print_copy_result(
        run_copy_microbench<Cfg8ch4fpga>("nt", copy_threads, duration_s));
    return 0;
  }

  if (mode != "ring" && mode != "direct") {
    std::cerr << "Unknown --mode: " << mode << std::endl;
    return 1;
  }

  std::cout << "bench_processor: config=" << config << " mode=" << mode
            << " duration=" << duration_s << "s"
            << " ring=" << BENCH_PROCESSOR_RING_BUFFER_SIZE
            << " workers=" << BENCH_PROCESSOR_WORKER_COUNT
            << " batch=" << SPATIAL_PROCESSOR_REGULAR_BATCH_SIZE
            << std::endl;

  // ProcessorState and its worker/feeder threads write chatter (startup,
  // shutdown, worker IDs) directly to std::cout.  Redirect cout to /dev/null
  // around each run so only the printf-based result lines reach the terminal.
  auto run_silent = [](auto fn) {
    std::ofstream devnull("/dev/null");
    auto *saved = std::cout.rdbuf(devnull.rdbuf());
    auto r = fn();
    std::cout.rdbuf(saved);
    return r;
  };

  bool ran = false;
  auto maybe_run = [&](const std::string &name, auto ring_fn, auto direct_fn) {
    if (config == "all" || config == name) {
      if (mode == "ring") {
        print_result(run_silent(ring_fn));
      } else {
        print_result(run_silent(direct_fn));
      }
      ran = true;
    }
  };

  maybe_run("1x1",
            [&]{ return run_processor_bench<Cfg1ch1fpga> (duration_s); },
            [&]{ return run_direct_processor_bench<Cfg1ch1fpga> (duration_s); });
  maybe_run("8x1",
            [&]{ return run_processor_bench<Cfg8ch1fpga> (duration_s); },
            [&]{ return run_direct_processor_bench<Cfg8ch1fpga> (duration_s); });
  maybe_run("16x1",
            [&]{ return run_processor_bench<Cfg16ch1fpga>(duration_s); },
            [&]{ return run_direct_processor_bench<Cfg16ch1fpga>(duration_s); });
  maybe_run("24x1",
            [&]{ return run_processor_bench<Cfg24ch1fpga>(duration_s); },
            [&]{ return run_direct_processor_bench<Cfg24ch1fpga>(duration_s); });
  maybe_run("32x1",
            [&]{ return run_processor_bench<Cfg32ch1fpga>(duration_s); },
            [&]{ return run_direct_processor_bench<Cfg32ch1fpga>(duration_s); });
  maybe_run("8x4",
            [&]{ return run_processor_bench<Cfg8ch4fpga> (duration_s); },
            [&]{ return run_direct_processor_bench<Cfg8ch4fpga> (duration_s); });
  maybe_run("16x4",
            [&]{ return run_processor_bench<Cfg16ch4fpga>(duration_s); },
            [&]{ return run_direct_processor_bench<Cfg16ch4fpga>(duration_s); });
  maybe_run("32x4",
            [&]{ return run_processor_bench<Cfg32ch4fpga>(duration_s); },
            [&]{ return run_direct_processor_bench<Cfg32ch4fpga>(duration_s); });

  if (!ran) {
    std::cerr << "Unknown --config: " << config << std::endl;
    return 1;
  }

  return 0;
}
