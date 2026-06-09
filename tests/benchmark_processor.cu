// benchmark_processor.cu — measures process_all_available_packets() throughput
// and microbenchmarks each key optimization applied to the processing path.
// Run directly (not via ctest): ./tests/benchmark_processor

#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.hpp"
#include "support/synthetic_packets.hpp"

#include <chrono>
#include <cstdio>
#include <sys/time.h>
#include <unordered_map>

using BenchConfig = LambdaConfig<2,  // NR_CHANNELS
                                  1,  // NR_FPGA_SOURCES
                                  8,  // NR_TIME_STEPS_PER_PACKET
                                  32, // NR_RECEIVERS
                                  2,  // NR_POLARIZATIONS
                                  32, // NR_RECEIVERS_PER_PACKET
                                  10, // NR_PACKETS_FOR_CORRELATION
                                  1,  // NR_BEAMS
                                  32, // NR_PADDED_RECEIVERS
                                  32, // NR_PADDED_RECEIVERS_PER_BLOCK
                                  1   // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                                  >;

class NullPipeline : public GPUPipeline {
public:
    void execute_pipeline(FinalPacketData *pkt, const bool = false) override {
        if (state_) state_->release_buffer(pkt->buffer_index);
    }
    void dump_visibilities(const uint64_t = 0) override {}
};

static constexpr size_t RING_SIZE = 2000;
static constexpr size_t NR_BUFS = 3;

// One batch = one full buffer: NR_PACKETS_FOR_CORRELATION+1 packets per channel
// (matching FillOneBufferTest pattern — the extra packet triggers completion).
static constexpr int PKTS_PER_CHANNEL = BenchConfig::NR_PACKETS_FOR_CORRELATION + 1;
static constexpr int PKTS_PER_BATCH =
    PKTS_PER_CHANNEL * BenchConfig::NR_CHANNELS;

static void feed_batch(ProcessorStateBase &state, uint64_t base_sample) {
    for (int ch = 0; ch < (int)BenchConfig::NR_CHANNELS; ++ch) {
        // One backward-extension packet (packet_index -1)
        test_support::feed_lambda_packet<BenchConfig>(
            state,
            base_sample - BenchConfig::NR_TIME_STEPS_PER_PACKET,
            0, static_cast<uint16_t>(ch),
            [](int, int, int) { return std::complex<int8_t>(1, 1); },
            [](int, int) { return int16_t(1); });

        for (int pkt = 0; pkt < PKTS_PER_CHANNEL; ++pkt) {
            uint64_t sample = base_sample + pkt * BenchConfig::NR_TIME_STEPS_PER_PACKET;
            test_support::feed_lambda_packet<BenchConfig>(
                state, sample, 0, static_cast<uint16_t>(ch),
                [](int, int, int) { return std::complex<int8_t>(1, 1); },
                [](int, int) { return int16_t(1); });
        }
    }
}

int main() {
    using clock = std::chrono::high_resolution_clock;

    std::array<int64_t, 1> delays = {0};
    std::unordered_map<uint32_t, int> fpga_map;
    fpga_map[0] = 0;

    auto *state = new ProcessorState<BenchConfig, NR_BUFS, RING_SIZE>(
        BenchConfig::NR_PACKETS_FOR_CORRELATION,
        BenchConfig::NR_TIME_STEPS_PER_PACKET,
        0, delays, fpga_map);

    auto *pipeline = new NullPipeline();
    pipeline->set_state(state);
    state->set_pipeline(pipeline);
    state->synchronous_pipeline = true;

    static constexpr int WARMUP = 10;
    static constexpr int BENCH = 500;

    // Each iteration's base_sample advances by PKTS_PER_CHANNEL time steps.
    const uint64_t step = PKTS_PER_CHANNEL * BenchConfig::NR_TIME_STEPS_PER_PACKET;

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        feed_batch(*state, 1000 + i * step);
        state->process_all_available_packets();
        state->handle_buffer_completion(true);
    }

    uint64_t before = state->packets_processed.load();
    auto t0 = clock::now();

    for (int i = 0; i < BENCH; ++i) {
        uint64_t base = 1000 + (WARMUP + i) * step;
        feed_batch(*state, base);
        state->process_all_available_packets();
        state->handle_buffer_completion(true);
    }

    auto t1 = clock::now();
    uint64_t processed = state->packets_processed.load() - before;
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double pps = processed / (elapsed_ms / 1000.0);

    size_t pkt_bytes = sizeof(EthernetHeader) + sizeof(IPHeader) +
                       sizeof(UDPHeader) + sizeof(CustomHeader) +
                       sizeof(BenchConfig::PacketPayloadType);

    printf("=== process_all_available_packets throughput ===\n");
    printf("  Config: %zu ch, 1 FPGA, %zu rx, %zu t/pkt\n",
           (size_t)BenchConfig::NR_CHANNELS,
           (size_t)BenchConfig::NR_RECEIVERS,
           (size_t)BenchConfig::NR_TIME_STEPS_PER_PACKET);
    printf("  Packet size: %zu bytes  Ring: %zu slots\n", pkt_bytes, RING_SIZE);
    printf("  Batches: %d  Packets/batch: %d\n", BENCH, PKTS_PER_BATCH);
    printf("  Processed: %lu pkts  Elapsed: %.1f ms\n", processed, elapsed_ms);
    printf("  Throughput: %.0f pkts/sec  payload %.1f MB/s\n",
           pps, pps * pkt_bytes / 1e6);

    // -----------------------------------------------------------------------
    // Microbenchmark A: hash lookup  (before: count+[] vs after: find)
    // -----------------------------------------------------------------------
    {
        std::unordered_map<uint32_t, int> m;
        for (int i = 0; i < 8; ++i) m[i] = i;
        constexpr int N = 20'000'000;
        volatile int sink = 0;

        auto ta = clock::now();
        for (int i = 0; i < N; ++i) {
            uint32_t k = i & 7;
            if (m.count(k)) sink += m[k]; // two traversals (before)
        }
        auto tb = clock::now();
        double before_ns = std::chrono::duration<double, std::nano>(tb - ta).count() / N;

        auto tc = clock::now();
        for (int i = 0; i < N; ++i) {
            uint32_t k = i & 7;
            auto it = m.find(k);
            if (it != m.end()) sink += it->second; // one traversal (after)
        }
        auto td = clock::now();
        double after_ns = std::chrono::duration<double, std::nano>(td - tc).count() / N;

        printf("\n=== Microbench A: hash lookup (N=%d) ===\n", N);
        printf("  count+[] (before): %.1f ns/call\n", before_ns);
        printf("  find     (after):  %.1f ns/call\n", after_ns);
        printf("  Savings: %.1f ns/pkt  At 1 Mpps: %.0f ms/sec CPU\n",
               before_ns - after_ns, (before_ns - after_ns));
        (void)sink;
    }

    // -----------------------------------------------------------------------
    // Microbenchmark B: gettimeofday cost (removed from add_received_packet_metadata)
    // -----------------------------------------------------------------------
    {
        constexpr int N = 5'000'000;
        struct timeval tv;
        auto ta = clock::now();
        for (int i = 0; i < N; ++i) gettimeofday(&tv, nullptr);
        auto tb = clock::now();
        double gtod_ns = std::chrono::duration<double, std::nano>(tb - ta).count() / N;

        printf("\n=== Microbench B: gettimeofday (N=%d) ===\n", N);
        printf("  Cost: %.1f ns/call\n", gtod_ns);
        printf("  Removed from hot path — at 1 Mpps saves: ~%.0f ms/sec CPU\n",
               gtod_ns);
    }

    // -----------------------------------------------------------------------
    // Microbenchmark C: pool alloc vs scattered alloc pointer-dereference cost
    // -----------------------------------------------------------------------
    {
        constexpr int N_SLOTS = 5000;
        constexpr int N_PASSES = 2000;

        // Scattered: N_SLOTS separate heap allocs (before)
        struct Slot { char data[512]; };
        std::vector<Slot*> scattered(N_SLOTS);
        for (int i = 0; i < N_SLOTS; ++i) scattered[i] = new Slot();

        // Contiguous pool (after)
        Slot *pool = new Slot[N_SLOTS]();
        std::vector<Slot*> pooled(N_SLOTS);
        for (int i = 0; i < N_SLOTS; ++i) pooled[i] = &pool[i];

        // Cold-cache sequential read pass
        volatile char sink = 0;
        auto ta = clock::now();
        for (int p = 0; p < N_PASSES; ++p)
            for (int i = 0; i < N_SLOTS; ++i) sink ^= scattered[i]->data[0];
        auto tb = clock::now();
        double scattered_ns = std::chrono::duration<double, std::nano>(tb - ta).count()
                              / (N_PASSES * N_SLOTS);

        auto tc = clock::now();
        for (int p = 0; p < N_PASSES; ++p)
            for (int i = 0; i < N_SLOTS; ++i) sink ^= pooled[i]->data[0];
        auto td = clock::now();
        double pooled_ns = std::chrono::duration<double, std::nano>(td - tc).count()
                           / (N_PASSES * N_SLOTS);

        printf("\n=== Microbench C: pointer deref — scattered vs pool (%d slots, %d passes) ===\n",
               N_SLOTS, N_PASSES);
        printf("  Scattered heap (before): %.2f ns/slot\n", scattered_ns);
        printf("  Contiguous pool (after): %.2f ns/slot\n", pooled_ns);
        printf("  Speedup: %.2fx  Savings: %.2f ns/slot\n",
               scattered_ns / pooled_ns, scattered_ns - pooled_ns);

        for (int i = 0; i < N_SLOTS; ++i) delete scattered[i];
        delete[] pool;
        (void)sink;
    }

    delete state;
    delete pipeline;
    return 0;
}
