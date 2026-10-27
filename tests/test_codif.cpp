#include "support/codif_packets.hpp"
#include <gtest/gtest.h>

using CodifConfig = LambdaConfig<8, 1, 64, 10, 2, 10, 8, 1, 32, 32, 1>;

static CodifConfig::PacketEntryType captured_packet() {
  const auto bytes = test_support::codif_packets().front();
  CodifConfig::PacketEntryType entry;
  std::memcpy(entry.data, bytes.data(), bytes.size());
  entry.length = bytes.size();
  entry.timestamp = {1234, 567};
  entry.sender_addr = {};
  std::memcpy(&entry.sender_addr.sin_addr.s_addr, bytes.data() + 26, 4);
  return entry;
}

TEST(CodifPacketTests, StandardMetadataKeepsSamplesScalesAndIpRouting) {
  for (bool bare : {false, true}) {
    for (auto format : {PacketFormat::Auto, PacketFormat::Codif}) {
      auto entry = captured_packet();
      const auto count = entry.parse().sample_count;
      const auto samples = std::vector<uint8_t>(entry.data + 106, entry.data + entry.length);
      test_support::standard_codif_header(entry.data + 42, 2);
      // Routing remains source-IP based, even if a header disagrees.
      entry.sender_addr.sin_addr.s_addr = htonl(0x0a00070a);
      if (bare) {
        entry.length -= 42;
        std::memmove(entry.data, entry.data + 42, entry.length);
      }
      const auto parsed = entry.parse(format);
      ASSERT_TRUE(parsed.valid());
      EXPECT_EQ(parsed.sample_count, count);
      EXPECT_EQ(parsed.freq_channel, 110);
      EXPECT_EQ(parsed.fpga_id, 7);
      EXPECT_EQ(std::memcmp(parsed.sample_data(), samples.data(), samples.size()), 0);
      CodifConfig::PacketScaleStructure scales;
      parsed.copy_scales(scales);
      for (auto &receiver : scales)
        for (auto scale : receiver) EXPECT_EQ(scale, 128);
    }
  }
}

TEST(CodifPacketTests, ValidatesMetadataLayoutAndAbsoluteThreadRange) {
  for (auto format : {PacketFormat::Auto, PacketFormat::Codif}) {
    for (uint16_t id : {0, 0x03bb, 1, 0xbb03, 0xffff}) {
      auto entry = captured_packet();
      test_support::codif_u16(entry.data + 42 + 44, id);
      EXPECT_EQ(entry.parse(format).valid(), id == 0 || id == 0x03bb);
      if (id != 0 && id != 0x03bb) {
        uint64_t original = 0;
        EXPECT_FALSE(prepare_replay_packet(entry.data, entry.length, 64, original, 3, format));
      }
    }
    for (uint16_t channel : {0, 511, 512, 65535}) {
      auto entry = captured_packet();
      test_support::standard_codif_header(entry.data + 42, 0);
      test_support::codif_u16(entry.data + 42 + 16, channel);
      const auto parsed = entry.parse(format);
      EXPECT_EQ(parsed.valid(), channel < 512);
      if (parsed.valid()) EXPECT_EQ(parsed.freq_channel, channel);
    }
  }
}

TEST(CodifPacketTests, StandardReplayKeepsGroupAndSourceIpTogether) {
  for (auto format : {PacketFormat::Auto, PacketFormat::Codif}) {
    auto entry = captured_packet();
    auto *header = entry.data + 42;
    test_support::standard_codif_header(header, entry.data[28]);
    test_support::codif_u32(header, codif::frames_per_period - 1);
    const auto count = entry.parse(format).sample_count;
    const auto metadata_and_samples = std::vector<uint8_t>(header + 44, entry.data + entry.length);
    uint64_t original = 0;
    ASSERT_TRUE(prepare_replay_packet(entry.data, entry.length, 128, original, 3, format));
    EXPECT_EQ(original, count);
    EXPECT_EQ(codif::u16(header + 18), 3);
    EXPECT_EQ(entry.data[28], 3);
    EXPECT_EQ(codif::u16(header + 20), 0);
    EXPECT_EQ(header[22], 'A');
    EXPECT_EQ(header[23], 't');
    EXPECT_EQ(std::memcmp(header + 44, metadata_and_samples.data(), metadata_and_samples.size()), 0);
    std::memcpy(&entry.sender_addr.sin_addr.s_addr, entry.data + 26, 4);
    EXPECT_EQ(entry.parse(format).sample_count, count + 128);
    EXPECT_EQ(entry.parse(format).fpga_id, 3);
    // A regular loop advances time without changing board identity.
    ASSERT_TRUE(prepare_replay_packet(entry.data, entry.length, 64, original, -1, format));
    EXPECT_EQ(codif::u16(header + 18), 3);
    EXPECT_EQ(entry.data[28], 3);
  }
}

TEST(CodifPacketTests, ParsesSuppliedCaptureWithoutChangingSamples) {
  const auto packets = test_support::codif_packets();
  uint64_t first = 0;
  for (size_t i = 0; i < packets.size(); ++i) {
    auto entry = captured_packet();
    std::memcpy(entry.data, packets[i].data(), packets[i].size());
    const auto parsed = entry.parse();
    ASSERT_TRUE(parsed.valid());
    ASSERT_NE(parsed.codif_header, nullptr);
    if (i == 0) first = parsed.sample_count;
    EXPECT_EQ(parsed.sample_count, first + (i / 8) * 64);
    EXPECT_EQ(parsed.freq_channel, 110 + i % 8);
    EXPECT_EQ(parsed.fpga_id, packets[i][28]);
    EXPECT_EQ(parsed.timestamp, 1234000567);
    EXPECT_EQ(parsed.payload_size, 2560);
    EXPECT_EQ(std::memcmp(parsed.sample_data(), packets[i].data() + 106, 2560), 0);
    CodifConfig::PacketScaleStructure scales;
    parsed.copy_scales(scales);
    for (auto &receiver : scales)
      for (auto scale : receiver) EXPECT_EQ(scale, 128);
  }
}

TEST(CodifPacketTests, SupportsBareUdpAndIpFpgaOverride) {
  using IpConfig = LambdaConfig<8, 1, 64, 10, 2, 10, 8, 1, 32, 32, 1, true>;
  auto source = captured_packet();
  IpConfig::PacketEntryType entry;
  entry.length = source.length - 42;
  entry.timestamp = {};
  entry.sender_addr = {};
  entry.sender_addr.sin_addr.s_addr = htonl(0x0a00070a);
  std::memcpy(entry.data, source.data + 42, entry.length);
  const auto parsed = entry.parse();
  ASSERT_TRUE(parsed.valid());
  EXPECT_EQ(parsed.fpga_id, 7);
  EXPECT_EQ(parsed.freq_channel, 110);
  EXPECT_EQ(parsed.sample_count, source.parse().sample_count);
}

TEST(CodifPacketTests, ExpandsAdcNibblesAndClampsLikeFirmware) {
  auto entry = captured_packet();
  for (int i = 0; i < 10; ++i)
    entry.data[42 + 46 + i] = ((2 * i) % 16) | (((2 * i + 1) % 16) << 4);
  const auto before = std::vector<uint8_t>(entry.data, entry.data + entry.length);
  CodifConfig::PacketScaleStructure scales;
  entry.parse().copy_scales(scales);
  for (int adc = 0; adc < 20; ++adc)
    EXPECT_EQ(scales[adc / 2][adc % 2], 1 << std::min(adc % 16, 8));
  EXPECT_EQ(std::memcmp(entry.data, before.data(), before.size()), 0);
}

TEST(CodifPacketTests, RejectsEveryTruncationAndMalformedGeometry) {
  for (bool bare : {false, true}) {
    auto entry = captured_packet();
    if (bare) {
      entry.length -= 42;
      std::memmove(entry.data, entry.data + 42, entry.length);
    }
    const int complete = entry.length;
    for (int length = 0; length < complete; ++length) {
      entry.length = length;
      entry.processed.store(false);
      EXPECT_FALSE(entry.parse().valid()) << "bare=" << bare << " length=" << length;
      EXPECT_TRUE(entry.processed.load());
    }
  }
  for (int offset : {9, 10, 14, 24, 26, 28, 32, 40}) {
    auto entry = captured_packet();
    entry.data[42 + offset] ^= 1;
    EXPECT_FALSE(entry.parse().valid()) << "header offset=" << offset;
  }
  for (int flags : {0x00, 0x05, 0x06, 0x14, 0x84}) {
    auto entry = captured_packet();
    entry.data[52] = flags;
    EXPECT_FALSE(entry.parse().valid()) << "flags=" << flags;
  }
  auto entry = captured_packet();
  test_support::codif_u32(entry.data + 42, codif::frames_per_period);
  EXPECT_FALSE(entry.parse().valid());
  entry.length = entry.DATA_CAPACITY + 1;
  EXPECT_FALSE(entry.parse().valid());
}

TEST(CodifPacketTests, FrameAndReferenceEpochRolloversAdvanceBy64Samples) {
  auto entry = captured_packet();
  auto *header = entry.data + 42;
  // 2025 July -> 2026 January, at the last alignment period of the old epoch.
  header[8] = 51;
  test_support::codif_u32(header + 4, 184 * 86400 - 27);
  test_support::codif_u32(header, codif::frames_per_period - 1);
  const auto last = entry.parse().sample_count;
  codif::advance(header, 64);
  EXPECT_EQ(entry.parse().sample_count, last + 64);
  EXPECT_EQ(codif::u32(header), 0);
  header[8] = 52;
  test_support::codif_u32(header + 4, 0);
  EXPECT_EQ(entry.parse().sample_count, last + 64);
  EXPECT_EQ(codif::epoch_seconds(0), 0);
  EXPECT_EQ(codif::epoch_seconds(1), 182 * 86400ULL);
  EXPECT_EQ(codif::epoch_seconds(201) - codif::epoch_seconds(200), 181 * 86400ULL);
}

TEST(CodifPacketTests, ReplayPreservesMetadataAndAdvancesAcrossPeriod) {
  auto entry = captured_packet();
  test_support::codif_u32(entry.data + 42, codif::frames_per_period - 1);
  const auto count = entry.parse().sample_count;
  const auto data = std::vector<uint8_t>(entry.data + 42 + 46, entry.data + entry.length);
  uint64_t original = 0;
  ASSERT_TRUE(prepare_replay_packet(entry.data, entry.length, 128, original, 3));
  EXPECT_EQ(original, count);
  EXPECT_EQ(entry.parse().sample_count, count + 128);
  EXPECT_EQ(codif::u16(entry.data + 42 + 18), 0xfeca);
  std::memcpy(&entry.sender_addr.sin_addr.s_addr, entry.data + 26, 4);
  EXPECT_EQ(entry.parse().fpga_id, 3);
  EXPECT_EQ(entry.parse().freq_channel, 110);
  EXPECT_EQ(std::memcmp(entry.data + 42 + 46, data.data(), data.size()), 0);
  EXPECT_THROW(prepare_replay_packet(entry.data, entry.length, 1, original), std::runtime_error);
}

TEST(CodifPacketTests, ExplicitParserAcceptsEpochZeroAndRejectsLegacy) {
  CodifPacketEntry<CodifConfig::PacketScaleStructure, CodifConfig::PacketDataStructure> entry;
  const auto packet = captured_packet();
  entry.length = packet.length;
  entry.timestamp = {};
  entry.sender_addr = {};
  std::memcpy(entry.data, packet.data, packet.length);
  entry.data[42 + 8] = 0;
  test_support::codif_u32(entry.data + 42 + 4, 0);
  ASSERT_TRUE(entry.parse().valid());
  EXPECT_EQ(entry.parse().sample_count, 0);
  std::memset(entry.data, 0, sizeof(entry.data));
  EXPECT_FALSE(entry.parse().valid());
}

TEST(CodifPacketTests, LegacyReplayStillUpdatesOnlyCounterAndFpgaId) {
  CodifConfig::PacketEntryType entry;
  entry.length = 42 + sizeof(CustomHeader) + sizeof(CodifConfig::PacketPayloadType);
  std::memset(entry.data, 0, sizeof(entry.data));
  CustomHeader header{640, 2, 117, {}};
  std::memcpy(entry.data + 42, &header, sizeof(header));
  uint64_t original = 0;
  ASSERT_TRUE(prepare_replay_packet(entry.data, entry.length, 256, original, 3));
  std::memcpy(&header, entry.data + 42, sizeof(header));
  EXPECT_EQ(original, 640);
  EXPECT_EQ(header.sample_count, 896);
  EXPECT_EQ(header.fpga_id, 3);
  EXPECT_EQ(header.freq_channel, 117);
}

TEST(CodifPacketTests, FixedModesAcceptOnlySelectedFormatInBothFramings) {
  for (bool bare : {false, true}) {
    auto entry = captured_packet();
    if (bare) {
      entry.length -= 42;
      std::memmove(entry.data, entry.data + 42, entry.length);
    }
    const auto expected = entry.parse();
    const auto parsed = entry.parse(PacketFormat::Codif);
    ASSERT_TRUE(parsed.valid());
    EXPECT_EQ(parsed.sample_count, expected.sample_count);
    EXPECT_EQ(parsed.freq_channel, expected.freq_channel);
    EXPECT_EQ(parsed.fpga_id, expected.fpga_id);
    EXPECT_EQ(parsed.sample_data(), expected.sample_data());
    EXPECT_FALSE(entry.parse(PacketFormat::Lambda).valid());
    entry.data[(bare ? 0 : 42) + 40] ^= 1;
    EXPECT_FALSE(entry.parse(PacketFormat::Codif).valid());

    std::memset(entry.data, 0, sizeof(entry.data));
    const size_t offset = bare ? 0 : 42;
    entry.length = offset + sizeof(CustomHeader) + sizeof(CodifConfig::PacketPayloadType);
    const CustomHeader header{640, 2, 117, {}};
    std::memcpy(entry.data + offset, &header, sizeof(header));
    const auto legacy = entry.parse(PacketFormat::Lambda);
    ASSERT_TRUE(legacy.valid());
    EXPECT_EQ(legacy.sample_count, 640);
    EXPECT_EQ(legacy.fpga_id, 2);
    EXPECT_EQ(legacy.freq_channel, 117);
    EXPECT_EQ(legacy.codif_header, nullptr);
    EXPECT_FALSE(entry.parse(PacketFormat::Codif).valid());
    --entry.length;
    EXPECT_FALSE(entry.parse(PacketFormat::Lambda).valid());
  }
}

TEST(CodifPacketTests, ReplayUsesSelectedFormatWithoutFallback) {
  auto entry = captured_packet();
  const auto count = entry.parse().sample_count;
  uint64_t original = 0;
  const auto before = std::vector<uint8_t>(entry.data, entry.data + entry.length);
  EXPECT_FALSE(prepare_replay_packet(entry.data, entry.length, 64, original, 3,
                                    PacketFormat::Lambda));
  EXPECT_EQ(std::memcmp(entry.data, before.data(), before.size()), 0);
  ASSERT_TRUE(prepare_replay_packet(entry.data, entry.length, 64, original, 3,
                                   PacketFormat::Codif));
  EXPECT_EQ(original, count);
  EXPECT_EQ(entry.parse(PacketFormat::Codif).sample_count, count + 64);

  std::memset(entry.data, 0, sizeof(entry.data));
  entry.length = 42 + sizeof(CustomHeader) + sizeof(CodifConfig::PacketPayloadType);
  const CustomHeader header{640, 2, 117, {}};
  std::memcpy(entry.data + 42, &header, sizeof(header));
  EXPECT_FALSE(prepare_replay_packet(entry.data, entry.length, 64, original, 3,
                                    PacketFormat::Codif));
  ASSERT_TRUE(prepare_replay_packet(entry.data, entry.length, 64, original, 3,
                                   PacketFormat::Lambda));
  EXPECT_EQ(original, 640);
  const auto parsed = entry.parse(PacketFormat::Lambda);
  EXPECT_EQ(parsed.sample_count, 704);
  EXPECT_EQ(parsed.fpga_id, 3);
}
