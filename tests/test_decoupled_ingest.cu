#include "spatial/spatial.hpp"
#include "support/synthetic_packets.hpp"
#include "support/pipeline_harness.hpp"
#include <gtest/gtest.h>
#include <functional>

namespace {
using Cfg = LambdaConfig<2, 4, 64, 40, 2, 10, 8, 1, 64, 32, 1>;
using State = ProcessorState<Cfg, 3, 512, 4>;
constexpr std::array<int64_t, 4> delays{0, 64, -64, 128};

bool wait_for(const std::function<bool()> &ready) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!ready() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  return ready();
}
std::complex<int8_t> value(int fp, int ch, int packet) {
  return {static_cast<int8_t>(1 + (packet + 1) % 61 + fp + ch),
          static_cast<int8_t>(-2 - (packet + 1) % 53 - fp - ch)};
}
int16_t scale(int fp, int ch, int packet) { return 2 + fp * 10 + ch + (packet + 1) % 7; }
bool hole(int fp, int ch, int packet) { return fp == 2 && ch == 1 && (packet == 3 || packet == 31); }

class DecoupledTest : public ::testing::Test {
protected:
  struct Sink : GPUPipeline {
    std::function<void(FinalPacketData *)> callback;
    void execute_pipeline(FinalPacketData *p, const bool = false) override { callback(p); }
    void dump_visibilities(const uint64_t = 0) override {}
  } sink;
  std::unique_ptr<State> state;
  std::thread coordinator;
  std::array<uint64_t, 4> linear{0, 1, 2, 3};
  std::atomic<bool> unblock{false};
  std::atomic<int> entered{0}, completed{0}, errors{0};
  uint64_t sent = 0;

  void SetUp() override {
    state = std::make_unique<State>(8, 64, 176, delays,
        std::unordered_map<uint32_t, int>{{0, 0}, {1, 1}, {2, 2}, {3, 3}});
    state->synchronous_pipeline = true;
    state->set_pipeline(&sink);
    // A known canonical origin lets us check every guard and calibrated FPGA
    // offset independently of which producer happens to win bootstrap.
    sockaddr_in addr{};
    const int length = test_support::build_constant_lambda_wire_packet<Cfg>(
        static_cast<uint8_t *>(state->get_current_write_pointer()), 640, 0, 176,
        value(0, 0, 0), scale(0, 0, 0));
    state->add_received_packet_metadata(length, addr);
    state->get_next_write_pointer();
    state->process_all_available_packets();
    ++sent;
    state->nr_capture_threads = 4;
  }
  void TearDown() override {
    unblock.store(true);
    state->shutdown();
    if (coordinator.joinable()) coordinator.join();
  }
  void start() { coordinator = std::thread([&] { state->process_packets(); }); }
  void send(int fp, int ch, int packet, bool corrupt = false) {
    void *ptr;
    int slot;
    ASSERT_EQ(state->reserve_write_batch_strided(fp, linear[fp], 1, &ptr, &slot), 1);
    const int length = test_support::build_constant_lambda_wire_packet<Cfg>(
        static_cast<uint8_t *>(ptr), 640 + packet * 64 + delays[fp], fp, 176 + ch,
        corrupt ? std::complex<int8_t>(100, -100) : value(fp, ch, packet),
        corrupt ? 100 : scale(fp, ch, packet));
    sockaddr_in addr{};
    state->commit_write_batch(1, &slot, &length, &addr);
    ++sent;
  }
  void send_range(int first, int last) {
    for (int packet = first; packet <= last; ++packet)
      for (int fp = 0; fp < 4; ++fp)
        for (int ch = 0; ch < 2; ++ch)
          if (!hole(fp, ch, packet) && !(fp == 0 && ch == 0 && packet == 0))
            send(fp, ch, packet);
  }
  void verify(FinalPacketData *base, int window) {
    auto *d = static_cast<Cfg::PacketFinalDataType *>(base);
    int bad = 0, missing = 0;
    bad += d->start_seq_id != 640 + window * 8 * 64;
    bad += d->end_seq_id != 640 + (window * 8 + 7) * 64;
    for (int p = 0; p < 10; ++p)
      for (int fp = 0; fp < 4; ++fp)
        for (int ch = 0; ch < 2; ++ch) {
          const int packet = window * 8 + p - 1;
          const bool absent = hole(fp, ch, packet);
          bad += d->arrivals[0][ch][p][fp] == absent;
          if (absent && p > 0 && p < 9) ++missing;
          for (int r = 0; r < 10; ++r)
            for (int pol = 0; pol < 2; ++pol) {
              bad += (*d->scales)[ch][p][fp * 10 + r][pol] !=
                  (absent ? 0 : scale(fp, ch, packet));
              // Missing input samples may be old bytes; zero scale masks them.
              if (!absent)
                for (int t = 0; t < 64; ++t)
                  bad += (*d->samples)[ch][p][fp][t][r][pol] != value(fp, ch, packet);
            }
        }
    bad += d->get_num_missing_packets() != missing;
    errors.fetch_add(bad);
  }
};

TEST_F(DecoupledTest, WorkersContinueWhileCompletionIsStalledAndSealedBufferIsImmutable) {
  sink.callback = [&](FinalPacketData *p) {
    const int window = entered.fetch_add(1);
    verify(p, window);
    if (window == 0) {
      while (!unblock.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      verify(p, window); // Stale corrupt packets must not alter the sealed buffer.
    }
    state->release_buffer(p->buffer_index);
    completed.fetch_add(1);
  };
  start();
  send_range(-1, 8);
  ASSERT_TRUE(wait_for([&] { return entered.load() == 1; }));
  send_range(9, 22); // Fill B and most of C while coordinator is blocked in A.
  send(0, 0, 2, true); // A late core packet cannot write A after seal.
  ASSERT_TRUE(wait_for([&] { return state->packets_processed.load() == sent; }));
  EXPECT_EQ(completed.load(), 0);
  EXPECT_GE(state->packets_discarded.load(), 1);
  for (int packet = 9; packet <= 22; ++packet)
    for (int fp = 0; fp < 4; ++fp)
      for (int ch = 0; ch < 2; ++ch) {
        auto *d = state->d_samples[packet / 8];
        EXPECT_EQ((*d->samples)[ch][packet % 8 + 1][fp][0][0][0], value(fp, ch, packet));
        EXPECT_EQ((*d->scales)[ch][packet % 8 + 1][fp * 10][0], scale(fp, ch, packet));
      }
  unblock.store(true);
  send_range(23, 64); // Reuse each physical slot repeatedly with distinct data.
  ASSERT_TRUE(wait_for([&] { return completed.load() == 8; }));
  EXPECT_EQ(errors.load(), 0);
  EXPECT_EQ(state->packets_missing, 2);
}

TEST_F(DecoupledTest, OutOfOrderReleasesPreserveGenerationsAndDeferredGuards) {
  state->future_stall_force_threshold = std::chrono::milliseconds(5);
  sink.callback = [&](FinalPacketData *p) {
    const int window = entered.fetch_add(1);
    verify(p, window);
    if (window >= 2) state->release_buffer(p->buffer_index);
    completed.fetch_add(1);
  };
  start();
  send_range(-1, 16);
  ASSERT_TRUE(wait_for([&] { return completed.load() == 2; }));
  state->release_buffer(1); // Window 4 is now open; window 3 is still unavailable.
  send_range(17, 40);
  ASSERT_TRUE(wait_for([&] { return state->packets_future_queued.load() >= 8; }));
  // Even timeout-forced watermarks cannot acknowledge a window while this
  // worker still owns deferred packets that belong to it.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(completed.load(), 2);
  state->release_buffer(0);
  ASSERT_TRUE(wait_for([&] { return completed.load() == 5; }));
  EXPECT_EQ(errors.load(), 0);
  EXPECT_EQ(state->packets_missing, 2);
}

TEST_F(DecoupledTest, StreamJumpRetiresEmptyWindowsWithoutDeadlockingFutureQueue) {
  state->future_stall_force_threshold = std::chrono::milliseconds(5);
  sink.callback = [&](FinalPacketData *p) {
    const int window = entered.fetch_add(1);
    if (window < 4) {
      if (p->get_num_missing_packets() != (window == 0 ? 63 : 64)) errors.fetch_add(1);
    } else {
      auto *d = static_cast<Cfg::PacketFinalDataType *>(p);
      if (d->get_num_missing_packets() != 0) errors.fetch_add(1);
      for (int packet = 32; packet < 40; ++packet)
        for (int fp = 0; fp < 4; ++fp)
          for (int ch = 0; ch < 2; ++ch)
            if ((*d->samples)[ch][packet % 8 + 1][fp][0][0][0] != value(fp, ch, packet))
              errors.fetch_add(1);
    }
    state->release_buffer(p->buffer_index);
    completed.fetch_add(1);
  };
  start();
  send_range(32, 40);
  ASSERT_TRUE(wait_for([&] { return completed.load() == 5; }));
  EXPECT_EQ(errors.load(), 0);
  EXPECT_EQ(state->packets_missing, 255);
}

TEST_F(DecoupledTest, ShutdownWithDeferredPacketsDoesNotNeedAReleasedGpuBuffer) {
  sink.callback = [&](FinalPacketData *) { completed.fetch_add(1); };
  start();
  send_range(-1, 32);
  ASSERT_TRUE(wait_for([&] { return state->packets_future_queued.load() >= 8; }));
  state->shutdown();
  coordinator.join();
  EXPECT_FALSE(state->running.load());
}

TEST_F(DecoupledTest, AsyncFeederFullGpuPipelineMatchesIndependentCpuCorrelation) {
  auto output = std::make_shared<SingleHostMemoryOutput<Cfg>>();
  auto pipeline = std::make_unique<LambdaStarspinPipeline<Cfg>>(2, 1000);
  pipeline->set_state(state.get());
  pipeline->set_output(output);
  std::array<int, 4> subpacket{};
  pipeline->set_subpacket_delays(subpacket.data());
  sink.callback = [&](FinalPacketData *p) {
    const int window = entered.fetch_add(1);
    verify(p, window);
    pipeline->execute_pipeline(p); // Actual asynchronous H2D release callback.
    pipeline->dump_visibilities();
    if (cudaDeviceSynchronize() != cudaSuccess) errors.fetch_add(1);
    // Independent reference: no ingest helpers, GPU kernels, or copied output.
    // Repository baseline convention is x_i * conj(x_j).
    for (int ch = 0; ch < 2; ++ch)
      for (int j = 0; j < 40; ++j)
        for (int i = 0; i <= j; ++i) {
          std::complex<double> expected{};
          for (int packet = window * 8; packet < (window + 1) * 8; ++packet) {
            if (hole(i / 10, ch, packet) || hole(j / 10, ch, packet)) continue;
            auto a = value(i / 10, ch, packet), b = value(j / 10, ch, packet);
            expected += 64.0 * std::complex<double>(a.real(), a.imag()) *
                std::conj(std::complex<double>(b.real(), b.imag())) *
                static_cast<double>(scale(i / 10, ch, packet) * scale(j / 10, ch, packet));
          }
          const int baseline = j * (j + 1) / 2 + i;
          for (int pol = 0; pol < 2; ++pol)
            for (int q = 0; q < 2; ++q)
              for (int component = 0; component < 2; ++component) {
                const double target = component ? expected.imag() : expected.real();
                const float actual = (*output->visibilities)[ch][baseline][pol][q][component];
                if (!std::isfinite(actual) || std::abs(actual - target) >
                    std::max(0.01, std::abs(target) * 2e-5)) {
                  if (errors.fetch_add(1) == 0)
                    ADD_FAILURE() << "window=" << window << " baseline=" << baseline
                                  << " component=" << component << " actual=" << actual
                                  << " expected=" << target;
                }
              }
        }
    completed.fetch_add(1);
  };
  state->synchronous_pipeline = false;
  std::thread feeder([&] { state->pipeline_feeder(); });
  start();
  send_range(-1, 32);
  const bool finished = wait_for([&] { return completed.load() == 4; });
  state->shutdown();
  coordinator.join();
  feeder.join();
  pipeline.reset(); // Join H2D reclaimer while the ProcessorState still exists.
  EXPECT_TRUE(finished);
  EXPECT_EQ(errors.load(), 0);
  EXPECT_EQ(state->packets_missing, 2);
}
} // namespace
