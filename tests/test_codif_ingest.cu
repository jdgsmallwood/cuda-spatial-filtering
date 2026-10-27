#include "spatial/spatial.hpp"
#include "support/codif_packets.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <thread>

static void check_codif_ingest(PacketFormat format) {
  using Config = LambdaConfig<2, 2, 64, 20, 2, 10, 16, 1, 32, 32, 1, true>;
  ProcessorState<Config, 3, 128, 2> state(16, 64, 110, {0, 64}, {{0, 0}, {1, 1}}, format);
  class Sink : public GPUPipeline {
    void execute_pipeline(FinalPacketData *, bool = false) override {
      ADD_FAILURE() << "Incomplete CODIF window was published";
    }
    void dump_visibilities(uint64_t = 0) override {}
  } sink;
  state.set_pipeline(&sink);
  state.synchronous_pipeline = true;
  state.nr_capture_threads = 2;
  const auto fixture = test_support::codif_packets().front();
  auto sample = [](int fp, int ch, int packet, int t, int adc) {
    return std::complex<int8_t>((fp * 31 + ch * 17 + packet + t + adc) % 127,
                               -1 - (packet + t + adc) % 126);
  };
  std::array<uint64_t, 2> linear{0, 1};
  for (int packet : {0, 2, 1, 3})
    for (int fp = 0; fp < 2; ++fp)
      for (int ch = 0; ch < 2; ++ch) {
        void *destination;
        int index;
        ASSERT_EQ(state.reserve_write_batch_strided(fp, linear[fp], 1, &destination, &index), 1);
        auto *bytes = static_cast<uint8_t *>(destination);
        std::memcpy(bytes, fixture.data(), fixture.size());
        auto *header = bytes + 42;
        if (format == PacketFormat::Codif)
          test_support::standard_codif_header(header, fp);
        test_support::codif_u32(header, codif::frames_per_period - 2);
        codif::advance(header, (packet + fp) * 64);
        test_support::codif_u16(header + 16, 110 + ch);
        for (int adc = 0; adc < 20; adc += 2)
          header[46 + adc / 2] = (adc % 9) | (((adc + 1) % 9) << 4);
        auto *samples = reinterpret_cast<Config::Sample *>(header + 64);
        for (int t = 0; t < 64; ++t)
          for (int adc = 0; adc < 20; ++adc)
            samples[t * 20 + adc] = sample(fp, ch, packet, t, adc);
        sockaddr_in sender{};
        sender.sin_addr.s_addr = htonl(0x0a00000a | (fp << 8));
        int length = fixture.size();
        state.commit_write_batch(1, &index, &length, &sender);
      }
  // Bootstrap the intended window before independent work ranges can race.
  state.packets_processed.fetch_add(state.process_work_range({0, 1, 1}));
  std::thread consumer([&] { state.process_packets(); });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (state.packets_processed.load() < 16 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  state.shutdown();
  consumer.join();
  ASSERT_EQ(state.packets_processed.load(), 16);
  ASSERT_EQ(state.packets_discarded.load(), 0);
  const auto &data = *state.d_samples[0];
  for (int ch = 0; ch < 2; ++ch)
    for (int packet = 0; packet < 4; ++packet)
      for (int fp = 0; fp < 2; ++fp) {
        ASSERT_TRUE((*data.arrivals)[ch][packet + 1][fp]);
        for (int adc = 0; adc < 20; ++adc) {
          ASSERT_EQ((*data.scales)[ch][packet + 1][fp * 10 + adc / 2][adc % 2], 1 << (adc % 9));
          for (int t = 0; t < 64; ++t)
            ASSERT_EQ((*data.samples)[ch][packet + 1][fp][t][adc / 2][adc % 2],
                      sample(fp, ch, packet, t, adc));
        }
      }
}

#ifdef SPATIAL_DIRECT_NIC_WORKERS
TEST(CodifDirectNicIngest, CopiesTwoFpgaStreamsWithDelaysAcrossFrameRollover) {
#else
TEST(CodifIngest, CopiesTwoFpgaStreamsWithDelaysAcrossFrameRollover) {
#endif
  check_codif_ingest(PacketFormat::Auto);
}

#ifdef SPATIAL_DIRECT_NIC_WORKERS
TEST(CodifDirectNicIngest, FixedFormatCopiesSamplesAndScales) {
#else
TEST(CodifIngest, FixedFormatCopiesSamplesAndScales) {
#endif
  check_codif_ingest(PacketFormat::Codif);
}
