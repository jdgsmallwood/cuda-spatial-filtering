#include "spatial/spatial.hpp"
#include "support/synthetic_packets.hpp"
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <complex>
#include <thread>

TEST(DirectNicWorkers, FourIndependentRingsPreservePacketSamples) {
  using Cfg = LambdaConfig<4, 4, 64, 40, 2, 10, 256, 1, 64, 32, 1>;
  ProcessorState<Cfg, 3, 256, 4> state(256, 64, 0, {},
                                      {{0, 0}, {1, 1}, {2, 2}, {3, 3}});
  class Sink : public GPUPipeline {
    void execute_pipeline(FinalPacketData *, const bool = false) override {
      ADD_FAILURE() << "Incomplete test window was published";
    }
    void dump_visibilities(const uint64_t = 0) override {}
  } sink;
  state.set_pipeline(&sink);
  state.synchronous_pipeline = true;

  constexpr uint64_t first_count = 640;
  const auto sample = [](int fp, int ch, int pkt, int t, int r, int p) {
    return std::complex<int8_t>((fp * 23 + ch * 11 + pkt + t + r + p) % 101,
                                -1 - (pkt + t * 3 + fp + r + p) % 101);
  };
  const auto fill = [&](void *dst, int fp, int ch, int pkt) {
    return test_support::build_lambda_wire_packet<Cfg>(
        static_cast<uint8_t *>(dst), first_count + pkt * 64, fp, ch,
        [&](int t, int r, int p) { return sample(fp, ch, pkt, t, r, p); },
        [&](int r, int p) {
          return static_cast<int16_t>(1 + fp * 100 + ch * 20 + r * 2 + p);
        });
  };
  sockaddr_in addr{};
  const int first_len = fill(state.get_current_write_pointer(), 0, 0, 0);
  state.add_received_packet_metadata(first_len, addr);
  state.get_next_write_pointer();
  state.process_all_available_packets();
  state.nr_capture_threads = 4;

  std::thread completion([&] { state.process_packets(); });
  std::array<std::thread, 4> producers;
  for (int fp = 0; fp < 4; ++fp) {
    producers[fp] = std::thread([&, fp] {
      uint64_t linear = fp;
      int flat = fp == 0 ? 1 : 0;
      while (flat < 4 * 128 && state.running.load()) {
        void *ptrs[31];
        int indices[31], lengths[31];
        sockaddr_in addrs[31]{};
        const int target = std::min(1 + flat % 31, 4 * 128 - flat);
        const int n = state.reserve_write_batch_strided(
            fp, linear, target, ptrs, indices);
        for (int i = 0; i < n; ++i)
          lengths[i] = fill(ptrs[i], fp, (flat + i) / 128, (flat + i) % 128);
        if (flat % 17 == 0)
          std::this_thread::sleep_for(std::chrono::microseconds(20));
        state.commit_write_batch(n, indices, lengths, addrs);
        flat += n;
      }
    });
  }

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(10);
  while (state.packets_processed.load() < 2048 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  state.shutdown();
  for (auto &producer : producers) producer.join();
  completion.join();

  ASSERT_EQ(state.packets_processed.load(), 2048);
  EXPECT_EQ(state.packets_discarded.load(), 0);
  const auto &data = *state.d_samples[0];
  for (int ch = 0; ch < 4; ++ch)
    for (int pkt = 0; pkt < 128; ++pkt)
      for (int fp = 0; fp < 4; ++fp) {
        ASSERT_TRUE(data.arrivals[0][ch][pkt + 1][fp]);
        for (int r = 0; r < 10; ++r)
          for (int p = 0; p < 2; ++p) {
            ASSERT_EQ((*data.scales)[ch][pkt + 1][fp * 10 + r][p],
                      1 + fp * 100 + ch * 20 + r * 2 + p);
            for (int t = 0; t < 64; ++t)
              ASSERT_EQ((*data.samples)[ch][pkt + 1][fp][t][r][p],
                        sample(fp, ch, pkt, t, r, p));
          }
      }
}

TEST(DirectNicWorkers, CompletesWindowsOnlyAfterAllFourRingsCopy) {
  using Cfg = LambdaConfig<1, 4, 64, 40, 2, 10, 8, 1, 64, 32, 1>;
  ProcessorState<Cfg, 3, 256, 4> state(8, 64, 0, {},
                                      {{0, 0}, {1, 1}, {2, 2}, {3, 3}});
  class Sink : public GPUPipeline {
  public:
    std::atomic<int> runs{0};
    std::atomic<int> bad_windows{0};
    std::atomic<int> bad_samples{0};
    void execute_pipeline(FinalPacketData *packet_data,
                          const bool = false) override {
      if (packet_data->get_num_missing_packets() != 0)
        bad_windows.fetch_add(1);
      // Check copied values at handoff, before recycling the host buffer.
      // This catches a completion barrier or batched-store fence publishing
      // an arrival flag before the corresponding samples are visible.
      const auto *data = static_cast<const Cfg::PacketFinalDataType *>(packet_data);
      bool bad = false;
      for (int pkt = 1; pkt <= 8; ++pkt)
        for (int fp = 0; fp < 4; ++fp)
          for (int r = 0; r < 10; ++r)
            for (int pol = 0; pol < 2; ++pol) {
              bad |= (*data->scales)[0][pkt][fp * 10 + r][pol] != 1;
              for (int t = 0; t < 64; ++t)
                bad |= (*data->samples)[0][pkt][fp][t][r][pol] !=
                    std::complex<int8_t>(fp + 1, -fp - 1);
            }
      if (bad) bad_samples.fetch_add(1);
      state_->release_buffer(packet_data->buffer_index);
      runs.fetch_add(1);
    }
    void dump_visibilities(const uint64_t = 0) override {}
  } sink;
  sink.set_state(&state);
  state.set_pipeline(&sink);
  state.synchronous_pipeline = true;
  state.nr_capture_threads = 4;

  std::thread completion([&] { state.process_packets(); });
  std::array<std::thread, 4> producers;
  for (int fp = 0; fp < 4; ++fp) {
    producers[fp] = std::thread([&, fp] {
      uint64_t linear = fp;
      for (int pkt = 0; pkt <= 24 && state.running.load(); ++pkt) {
        void *ptr;
        int slot;
        const int n = state.reserve_write_batch_strided(
            fp, linear, 1, &ptr, &slot);
        if (n != 1) break;
        const int length = test_support::build_constant_lambda_wire_packet<Cfg>(
            static_cast<uint8_t *>(ptr), 640 + pkt * 64, fp, 0,
            {static_cast<int8_t>(fp + 1), static_cast<int8_t>(-fp - 1)}, 1);
        if (fp == 3 && pkt % 3 == 0)
          std::this_thread::sleep_for(std::chrono::microseconds(30));
        sockaddr_in addr{};
        state.commit_write_batch(1, &slot, &length, &addr);
      }
    });
  }

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(5);
  while (sink.runs.load() < 3 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  state.shutdown();
  for (auto &producer : producers) producer.join();
  completion.join();

  EXPECT_EQ(sink.runs.load(), 3);
  EXPECT_EQ(sink.bad_windows.load(), 0);
  EXPECT_EQ(sink.bad_samples.load(), 0);
  EXPECT_EQ(state.packets_missing, 0);
}

TEST(DirectNicWorkers, FullCommittedRingDoesNotLookEmpty) {
  using Cfg = LambdaConfig<1, 4, 64, 40, 2, 10, 256, 1, 64, 32, 1>;
  ProcessorState<Cfg, 3, 256, 4> state(256, 64, 0, {},
                                      {{0, 0}, {1, 1}, {2, 2}, {3, 3}});
  class Sink : public GPUPipeline {
    void execute_pipeline(FinalPacketData *, const bool = false) override {
      ADD_FAILURE() << "Incomplete test window was published";
    }
    void dump_visibilities(const uint64_t = 0) override {}
  } sink;
  state.set_pipeline(&sink);
  state.synchronous_pipeline = true;
  state.nr_capture_threads = 4;

  uint64_t linear = 0;
  void *ptrs[64];
  int indices[64], lengths[64];
  sockaddr_in addrs[64]{};
  ASSERT_EQ(state.reserve_write_batch_strided(0, linear, 64, ptrs, indices),
            64);
  for (int i = 0; i < 64; ++i)
    lengths[i] = test_support::build_constant_lambda_wire_packet<Cfg>(
        static_cast<uint8_t *>(ptrs[i]), 640 + i * 64, 0, 0, {2, -2}, 1);
  state.commit_write_batch(64, indices, lengths, addrs);

  std::thread completion([&] { state.process_packets(); });
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(2);
  while (state.packets_processed.load() < 64 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  state.shutdown();
  completion.join();
  EXPECT_EQ(state.packets_processed.load(), 64);
}
