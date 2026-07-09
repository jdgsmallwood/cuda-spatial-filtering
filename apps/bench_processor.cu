#include <argparse/argparse.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sched.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "spatial/common.hpp"
#include "spatial/logging.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.hpp"
#include "synthetic_packets.hpp"
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>

// Pin the whole process (and therefore every thread it later spawns, which
// inherit the mask) to the CPUs of a single NUMA node.  This box is a
// dual-socket Xeon with a *split* L3 (one ~14 MB slice per socket) and separate
// memory controllers, so letting the scheduler scatter the producer / processor
// / feeder / worker threads across both sockets means every ring-slot read and
// every pinned-buffer write can cross the inter-socket link -- roughly halving
// throughput.  Confining everything to one node keeps the ring hot in that
// node's L3 and every access node-local (default first-touch policy then places
// all the allocations there too, since the constructor runs on this thread).
// Honours SPATIAL_BENCH_NODE (default 0); set to -1 to disable pinning.
static void pin_to_numa_node(int node) {
  if (node < 0)
    return;
  std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) +
                  "/cpulist");
  std::string cpulist;
  if (!std::getline(f, cpulist) || cpulist.empty())
    return;

  cpu_set_t set;
  CPU_ZERO(&set);
  // cpulist is a comma-separated list of ranges, e.g. "0-9" or "0-4,10-14".
  size_t i = 0;
  while (i < cpulist.size()) {
    size_t comma = cpulist.find(',', i);
    if (comma == std::string::npos)
      comma = cpulist.size();
    std::string token = cpulist.substr(i, comma - i);
    size_t dash = token.find('-');
    int lo = std::stoi(token.substr(0, dash));
    int hi = dash == std::string::npos ? lo : std::stoi(token.substr(dash + 1));
    for (int c = lo; c <= hi; ++c)
      CPU_SET(c, &set);
    i = comma + 1;
  }
  sched_setaffinity(0, sizeof(set), &set);
}

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
  uint64_t packets_future_queued;
  uint64_t packets_stuck;
  uint64_t buffers_completed;
  double elapsed;
  double packets_per_sec;
  double gb_per_sec;
};

// Local equivalent of pipeline/common.hpp's BufferReleaseContext.
// We can't include that header (it pulls in libtcc, ccglib, PSRDADA, etc.).
struct H2DReleaseContext {
  ProcessorStateBase *state;
  size_t buffer_index;
  std::atomic<uint64_t> *buffers_completed;
};

static void h2d_release_host_func(void *data) {
  auto *ctx = static_cast<H2DReleaseContext *>(data);
  ctx->buffers_completed->fetch_add(1, std::memory_order_relaxed);
  ctx->state->release_buffer(ctx->buffer_index);
  delete ctx;
}

// Stands in for the GPU pipeline.  Without --with-h2d it immediately hands
// the buffer straight back to the processor, so this benchmark measures only
// the capture-ring -> reassembly path.  With --with-h2d it performs an async
// H2D cudaMemcpyAsync of the samples buffer (source is cudaHostAlloc pinned
// memory — a true PCIe DMA) and releases via a cudaLaunchHostFunc callback
// after the transfer completes, mirroring LambdaGPUPipeline's ingest path.
class NullPipeline : public GPUPipeline {
  // One CUDA stream per input-buffer slot.  H2D copy and the release callback
  // are enqueued on the same stream so the callback always fires after the
  // transfer — no cross-stream events needed.  NUM_BUFS matches NR_INPUT_BUFFERS
  // used in run_processor_bench.
  static constexpr int NUM_BUFS = 8;
  void *d_bufs_[NUM_BUFS]{};
  cudaStream_t streams_[NUM_BUFS]{};
  const bool with_h2d_;

public:
  std::atomic<uint64_t> buffers_completed{0};

  explicit NullPipeline(bool with_h2d = false) : with_h2d_(with_h2d) {
    if (with_h2d_) {
      for (int i = 0; i < NUM_BUFS; ++i)
        CUDA_CHECK(
            cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking));
    }
  }

  ~NullPipeline() {
    if (with_h2d_) {
      for (int i = 0; i < NUM_BUFS; ++i) {
        // Synchronize before destroying: ensures the release callback has
        // called state_->release_buffer() before we return.  ProcessorState
        // outlives NullPipeline (reverse construction order in
        // run_processor_bench), so state_ is still valid during the callback.
        if (streams_[i]) {
          cudaStreamSynchronize(streams_[i]);
          cudaStreamDestroy(streams_[i]);
        }
        if (d_bufs_[i])
          cudaFree(d_bufs_[i]);
      }
    }
  }

  void execute_pipeline(FinalPacketData *packet_data,
                        const bool = false) override {
    if (with_h2d_) {
      const int idx = static_cast<int>(packet_data->buffer_index);
      // Lazy device-buffer allocation on first use (size only known here).
      if (!d_bufs_[idx])
        CUDA_CHECK(cudaMalloc(&d_bufs_[idx],
                              packet_data->get_samples_elements_size()));
      // Async H2D from pinned host (cudaHostAlloc in LambdaFinalPacketData
      // constructor) — a true PCIe DMA with no internal staging copy.
      CUDA_CHECK(cudaMemcpyAsync(d_bufs_[idx], packet_data->get_samples_ptr(),
                                 packet_data->get_samples_elements_size(),
                                 cudaMemcpyHostToDevice, streams_[idx]));
      // Release callback on the SAME stream — fires after H2D completes.
      auto *ctx =
          new H2DReleaseContext{.state = state_,
                                .buffer_index = packet_data->buffer_index,
                                .buffers_completed = &buffers_completed};
      CUDA_CHECK(
          cudaLaunchHostFunc(streams_[idx], h2d_release_host_func, ctx));
    } else {
      buffers_completed.fetch_add(1, std::memory_order_relaxed);
      state_->release_buffer(static_cast<int>(packet_data->buffer_index));
    }
  }

  void dump_visibilities(const uint64_t = 0) override {}
};

template <typename Config>
BenchResult run_processor_bench(double duration_s, bool with_h2d = false) {
  constexpr size_t PKT_BYTES =
      sizeof(EthernetHeader) + sizeof(IPHeader) + sizeof(UDPHeader) +
      sizeof(CustomHeader) + sizeof(typename Config::PacketPayloadType);

  std::array<int64_t, Config::NR_FPGA_SOURCES> fpga_delays{};
  std::unordered_map<uint32_t, int> fpga_ids;
  for (size_t i = 0; i < Config::NR_FPGA_SOURCES; ++i) {
    fpga_ids[static_cast<uint32_t>(i)] = static_cast<int>(i);
  }

  constexpr size_t NR_INPUT_BUFFERS = 8;
  // Ring sizing constraints (sweep in
  // scripts/profiling/BENCH_PROCESSOR_TUNING.md):
  //   1. Divisible by PRODUCER_COUNT -- strided slot ownership is only
  //      disjoint between producers when the stride divides the ring;
  //      otherwise producers share slots and stall on each other (the
  //      processor loop warns about this).
  //   2. RING / PRODUCER_COUNT <= the smallest config's buffer horizon
  //      (NR_INPUT_BUFFERS * (NR_PACKETS_FOR_CORRELATION+1) ~= 2056 packets
  //      for the 1-channel config) -- a producer that can run further ahead
  //      than the horizon floods the future queue and trips the stall
  //      safety net, force-completing buffers with holes.
  //   6144 = 3 * 2048 satisfies both.  Cache-locality matters less than in
  //   the single-producer round (6 striding workers prefetch well past L3).
  // 3 producers + 6 workers + the processor thread = 10 busy threads on this
  // 10-core-per-socket box (the feeder sleeps in synchronous mode).  More of
  // either oversubscribes the socket and collapses throughput (12 busy
  // threads measured ~0.7 GB/s: spin-waits get descheduled).
  constexpr int WORKER_COUNT = 6;
  ProcessorState<Config, NR_INPUT_BUFFERS, DEFAULT_PACKET_RING_BUFFER_SIZE,
                 WORKER_COUNT>
      state(Config::NR_PACKETS_FOR_CORRELATION,
            Config::NR_TIME_STEPS_PER_PACKET,
            /*min_freq_channel=*/0, fpga_delays, fpga_ids);

  NullPipeline pipeline(with_h2d);
  state.set_pipeline(&pipeline);
  pipeline.set_state(&state);

  // Synchronous mode: execute_pipeline() is called directly from
  // handle_buffer_completion() (not via the async SPSC queue), avoiding the
  // pipeline_feeder shutdown race where the feeder exits before draining
  // pipeline_q.  Without --with-h2d, release_buffer() fires inside
  // execute_pipeline so the cv wait in advance_to_next_buffer never blocks.
  // With --with-h2d, execute_pipeline enqueues the H2D + callback and returns
  // immediately; advance_to_next_buffer then blocks briefly (~microseconds for
  // 2.5 KB at PCIe speeds) until the cudaLaunchHostFunc callback fires and
  // calls release_buffer.
  state.synchronous_pipeline = true;

  // Multi-threaded strided producers (the same lock-free protocol live
  // multi-queue capture uses: reserve_write_batch_strided -> fill ->
  // commit_write_batch; producer t owns ring slots t, t+N, t+2N, ...).  A
  // single producer building 2664 B/packet runs in lockstep with the
  // reassembly workers (~one 2.2 GHz core of memcpy ≈ the consumer rate), so
  // batches stay small and fork/join overhead caps worker scaling; several
  // producers mirror real multi-queue NIC capture and let the workers see
  // full batches.  Correct since the completion-watermark fix in
  // copy_data_to_input_buffer_if_able (deferred packets no longer complete
  // buffers prematurely) -- see scripts/profiling/BENCH_PROCESSOR_TUNING.md.
  constexpr int PRODUCER_COUNT = 3;
  state.nr_capture_threads = PRODUCER_COUNT;

  const auto sample_fn = [](int, int, int) {
    return std::complex<int8_t>(2, -2);
  };
  const auto scale_fn = [](int, int) { return static_cast<int16_t>(1); };

  constexpr uint64_t NR_BETWEEN_SAMPLES = Config::NR_TIME_STEPS_PER_PACKET;
  const uint64_t start_sample = NR_BETWEEN_SAMPLES * 10;

  // Prebuild one wire-format template ONCE (shared, read-only across the
  // producers).  Per packet a producer just memcpy's it into its reserved
  // ring slot and patches the CustomHeader (sample_count/fpga/channel),
  // mirroring a real capture (NIC delivers wire bytes -> single memcpy)
  // instead of rebuilding 1280 samples with per-element lambda calls.
  std::vector<uint8_t> tmpl(PKT_BYTES);
  const size_t tmpl_len = test_support::build_lambda_wire_packet<Config>(
      tmpl.data(), /*sample_count=*/1, /*fpga_id=*/0, /*freq_channel=*/0,
      sample_fn, scale_fn);
  constexpr size_t CUSTOM_OFF =
      sizeof(EthernetHeader) + sizeof(IPHeader) + sizeof(UDPHeader);

  std::thread processor([&state]() { state.process_packets(); });
  std::thread feeder([&state]() { state.pipeline_feeder(); });

  // Bounded-skew throttle.  Real FPGA streams stay roughly sequence-aligned
  // over time, and the reassembler's slack (NR_INPUT_BUFFERS windows plus the
  // half-window completion margin) absorbs bounded skew -- but not unbounded
  // drift, so producers must not run away from each other.  Each producer
  // publishes its completed-round count; nobody starts round R until the
  // slowest producer has completed round R - MAX_ROUND_SKEW.  Unlike the
  // hard per-round barrier this doesn't serialise the producers, it only
  // clamps their drift.
  constexpr uint64_t MAX_ROUND_SKEW = 2;
  struct alignas(64) RoundCounter {
    std::atomic<uint64_t> v{0};
  };
  std::array<RoundCounter, PRODUCER_COUNT> completed_rounds;

  std::array<uint64_t, PRODUCER_COUNT> fed_counts{};

  // Producer `tid` feeds every (channel, fpga) stream whose flat index is
  // congruent to tid mod PRODUCER_COUNT, so each stream's packets stay
  // in-order within one producer.  Batches of descriptors are staged and
  // flushed through reserve/fill/commit so the ring sees large contiguous
  // claims.
  auto producer_fn = [&](int tid) {
    constexpr int RBATCH = 128;
    void *slot_ptrs[RBATCH];
    int slot_indices[RBATCH];
    int lens[RBATCH];
    sockaddr_in addrs[RBATCH];
    sockaddr_in base_addr{};
    base_addr.sin_family = AF_INET;
    base_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    uint64_t sc_buf[RBATCH];
    uint32_t fp_buf[RBATCH];
    uint16_t ch_buf[RBATCH];
    int pend = 0;
    uint64_t my_linear = static_cast<uint64_t>(tid);
    uint64_t fed = 0;

    auto flush = [&]() {
      int done = 0;
      while (done < pend) {
        const int got = state.reserve_write_batch_strided(
            tid, my_linear, pend - done, slot_ptrs, slot_indices);
        if (got <= 0) {
          if (!state.running.load(std::memory_order_acquire)) {
            pend = 0;
            return;
          }
          continue;
        }
        for (int i = 0; i < got; ++i) {
          uint8_t *dst = reinterpret_cast<uint8_t *>(slot_ptrs[i]);
          std::memcpy(dst, tmpl.data(), tmpl_len);
          CustomHeader *custom =
              reinterpret_cast<CustomHeader *>(dst + CUSTOM_OFF);
          custom->sample_count = sc_buf[done + i];
          custom->fpga_id = fp_buf[done + i];
          custom->freq_channel = ch_buf[done + i];
          lens[i] = static_cast<int>(tmpl_len);
          addrs[i] = base_addr;
        }
        state.commit_write_batch(got, slot_indices, lens, addrs);
        done += got;
      }
      pend = 0;
    };
    auto emit = [&](uint64_t sc, uint32_t fp, uint16_t ch) {
      sc_buf[pend] = sc;
      fp_buf[pend] = fp;
      ch_buf[pend] = ch;
      ++pend;
      ++fed;
      if (pend == RBATCH)
        flush();
    };

    uint64_t round = 0;
    while (state.running.load(std::memory_order_acquire)) {
      // Bounded-skew wait: don't start round R until the slowest producer
      // has completed round R - MAX_ROUND_SKEW.
      while (state.running.load(std::memory_order_acquire)) {
        uint64_t min_done = UINT64_MAX;
        for (int j = 0; j < PRODUCER_COUNT; ++j) {
          min_done = std::min(
              min_done, completed_rounds[j].v.load(std::memory_order_acquire));
        }
        if (round <= min_done + MAX_ROUND_SKEW)
          break;
        _mm_pause();
      }

      const uint64_t round_start_sample =
          start_sample +
          round * Config::NR_PACKETS_FOR_CORRELATION * NR_BETWEEN_SAMPLES;
      int stream = 0;
      for (size_t channel = 0; channel < Config::NR_CHANNELS; ++channel) {
        for (size_t fpga = 0; fpga < Config::NR_FPGA_SOURCES;
             ++fpga, ++stream) {
          if (stream % PRODUCER_COUNT != tid)
            continue;
          for (int pkt = 0;
               pkt <= static_cast<int>(Config::NR_PACKETS_FOR_CORRELATION);
               ++pkt) {
            emit(round_start_sample +
                     static_cast<uint64_t>(pkt) * NR_BETWEEN_SAMPLES,
                 static_cast<uint32_t>(fpga), static_cast<uint16_t>(channel));
          }
          // The "-1" packet (one before the buffer start) is emitted last,
          // mirroring SyntheticPipelineRun's proven per-stream fill order.
          emit(round_start_sample - NR_BETWEEN_SAMPLES,
               static_cast<uint32_t>(fpga), static_cast<uint16_t>(channel));
        }
      }
      flush(); // publish this round's packets before advancing the counter
      ++round;
      completed_rounds[tid].v.store(round, std::memory_order_release);
    }
    flush();
    fed_counts[tid] = fed;
  };

  const auto start = std::chrono::steady_clock::now();
  std::vector<std::thread> producers;
  producers.reserve(PRODUCER_COUNT);
  for (int t = 0; t < PRODUCER_COUNT; ++t)
    producers.emplace_back(producer_fn, t);

  std::this_thread::sleep_for(std::chrono::duration<double>(duration_s));

  state.shutdown();
  for (auto &p : producers)
    p.join();
  processor.join();
  feeder.join();

  uint64_t packets_fed = 0;
  for (auto c : fed_counts)
    packets_fed += c;

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
  r.packets_future_queued = state.packets_future_queued.load();
  r.packets_stuck = state.packets_stuck_unprocessed.load();
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
      "packets_discarded=%llu future_queued=%llu stuck=%llu "
      "buffers_completed=%llu "
      "elapsed=%.3f bytes_per_packet=%zu "
      "packets/sec=%.2f GB/sec=%.6f\n",
      r.nr_channels, r.nr_fpga_sources, r.nr_receivers,
      (unsigned long long)r.packets_fed,
      (unsigned long long)r.packets_processed,
      (unsigned long long)r.packets_missing,
      (unsigned long long)r.packets_discarded,
      (unsigned long long)r.packets_future_queued,
      (unsigned long long)r.packets_stuck,
      (unsigned long long)r.buffers_completed, r.elapsed, r.bytes_per_packet,
      r.packets_per_sec, r.gb_per_sec);
  std::fflush(stdout);
}

int main(int argc, char *argv[]) {
  // Confine the benchmark to a single NUMA node before any allocation so the
  // ring/pinned buffers first-touch node-local memory and stay in one L3.
  const char *node_env = std::getenv("SPATIAL_BENCH_NODE");
  pin_to_numa_node(node_env ? std::atoi(node_env) : 0);

  argparse::ArgumentParser program("bench_processor");
  program.add_argument("--duration")
      .help("Duration in seconds to feed synthetic packets per config")
      .default_value(10.0)
      .scan<'g', double>();
  program.add_argument("--with-h2d")
      .help("Async H2D cudaMemcpyAsync of each samples buffer before release "
            "(verifies writes are not dead-stored; exercises PCIe bandwidth; "
            "mirrors LambdaGPUPipeline's ingest path)")
      .default_value(false)
      .implicit_value(true);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  const double duration_s = program.get<double>("--duration");
  const bool with_h2d = program.get<bool>("--with-h2d");

  // Redirect the default stdout spdlog logger to a file so INFO_LOG calls
  // from ProcessorState don't pollute the benchmark's stdout output.
  static auto tp = std::make_shared<spdlog::details::thread_pool>(8192, 1);
  auto logger = std::make_shared<spdlog::async_logger>(
      "bench_processor",
      std::make_shared<spdlog::sinks::basic_file_sink_mt>("app.log", false),
      tp, spdlog::async_overflow_policy::overrun_oldest);
  logger->set_level(spdlog::level::warn);
  spatial::Logger::set(logger);

  std::cout << "bench_processor: 8 configs x " << duration_s << "s each";
  if (with_h2d)
    std::cout << "  [--with-h2d: async H2D transfer + callback enabled]";
  std::cout << std::endl;

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

  print_result(run_silent([&]{ return run_processor_bench<Cfg1ch1fpga> (duration_s, with_h2d); }));
  print_result(run_silent([&]{ return run_processor_bench<Cfg8ch1fpga> (duration_s, with_h2d); }));
  print_result(run_silent([&]{ return run_processor_bench<Cfg16ch1fpga>(duration_s, with_h2d); }));
  print_result(run_silent([&]{ return run_processor_bench<Cfg24ch1fpga>(duration_s, with_h2d); }));
  print_result(run_silent([&]{ return run_processor_bench<Cfg32ch1fpga>(duration_s, with_h2d); }));
  print_result(run_silent([&]{ return run_processor_bench<Cfg8ch4fpga> (duration_s, with_h2d); }));
  print_result(run_silent([&]{ return run_processor_bench<Cfg16ch4fpga>(duration_s, with_h2d); }));
  print_result(run_silent([&]{ return run_processor_bench<Cfg32ch4fpga>(duration_s, with_h2d); }));

  return 0;
}
