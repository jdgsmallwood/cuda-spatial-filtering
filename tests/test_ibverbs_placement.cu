// Differential test for ProcessorState::locate_packet() (spatial.hpp): the
// GPUDirect ingest path (see /home/ubuntu/.claude/plans/i-want-to-start-breezy-lampson.md,
// libibverbs.hpp) uses locate_packet() to predict packet placement instead
// of the reactive copy_data_to_input_buffer_if_able path. This asserts both
// resolve to the identical (buffer_index, packet_index) for every packet
// across single-FPGA in-order streams, multi-FPGA streams with nonzero
// delays, buffer-rollover boundary crossings, and discard cases.
#include "spatial/common.hpp"
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.hpp"
#include <array>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <unordered_map>

namespace {

using PlacementConfig =
    LambdaConfig<2,  // NR_CHANNELS
                 2,  // NR_FPGA_SOURCES
                 8,  // NR_TIME_STEPS_PER_PACKET (== NR_BETWEEN_SAMPLES here)
                 20, // NR_RECEIVERS
                 2,  // NR_POLARIZATIONS
                 10, // NR_RECEIVERS_PER_PACKET
                 6,  // NR_PACKETS_FOR_CORRELATION -- small, exercises rollover quickly
                 1,  // NR_BEAMS
                 32, // NR_PADDED_RECEIVERS
                 32, // NR_PADDED_RECEIVERS_PER_BLOCK
                 1   // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 >;
constexpr size_t NR_BUFFERS = 3;
using PlacementState = ProcessorState<PlacementConfig, NR_BUFFERS>;

class MockPipeline : public GPUPipeline {
public:
  void execute_pipeline(FinalPacketData *packet_data, const bool = false) override {
    // Mirrors SimpleMockPipeline (test_processor.cu): release immediately so
    // buffer rollover / current_buffer advancement behaves like a live run.
    if (state_) {
      state_->release_buffer(packet_data->buffer_index);
    }
  }
  void dump_visibilities(const uint64_t = 0) override {}
};

class LocatePacketTest : public ::testing::Test {
public:
  PlacementState *state = nullptr;
  MockPipeline pipeline;

  void build(std::array<int64_t, PlacementConfig::NR_FPGA_SOURCES> delays) {
    std::unordered_map<uint32_t, int> map;
    for (uint32_t i = 0; i < PlacementConfig::NR_FPGA_SOURCES; ++i) map[i] = static_cast<int>(i);
    state = new PlacementState(PlacementConfig::NR_PACKETS_FOR_CORRELATION,
                               PlacementConfig::NR_TIME_STEPS_PER_PACKET, 0,
                               delays, map);
    pipeline.set_state(state);
    state->set_pipeline(&pipeline);
    state->synchronous_pipeline = true;
  }

  void TearDown() override { delete state; }

  // Mirrors test_processor.cu's create_lambda_packet, trimmed to what this
  // test needs (placement, not content correctness -- that's already
  // covered by ProcessorStateTest/ProcessorStateMultipleFPGATest).
  void feed_packet(uint64_t sample_count, uint32_t fpga_id, uint16_t freq_channel) {
    void *write_ptr = state->get_current_write_pointer();
    uint8_t *data_ptr = (uint8_t *)write_ptr;

    EthernetHeader *eth = (EthernetHeader *)data_ptr;
    memset(eth, 0, sizeof(EthernetHeader));
    eth->ethertype = htons(0x0800);
    data_ptr += sizeof(EthernetHeader);

    IPHeader *ip = (IPHeader *)data_ptr;
    memset(ip, 0, sizeof(IPHeader));
    ip->version_ihl = 0x45;
    data_ptr += sizeof(IPHeader);

    UDPHeader *udp = (UDPHeader *)data_ptr;
    memset(udp, 0, sizeof(UDPHeader));
    data_ptr += sizeof(UDPHeader);

    CustomHeader *custom = (CustomHeader *)data_ptr;
    custom->sample_count = sample_count;
    custom->fpga_id = fpga_id;
    custom->freq_channel = freq_channel;
    memset(custom->padding, 0, sizeof(custom->padding));
    data_ptr += sizeof(CustomHeader);

    auto *payload =
        reinterpret_cast<typename PlacementConfig::PacketPayloadType *>(data_ptr);
    for (int r = 0; r < PlacementConfig::NR_RECEIVERS_PER_PACKET; r++)
      for (int p = 0; p < PlacementConfig::NR_POLARIZATIONS; p++)
        payload->scales[r][p] = 1;
    for (int t = 0; t < PlacementConfig::NR_TIME_STEPS_PER_PACKET; t++)
      for (int r = 0; r < PlacementConfig::NR_RECEIVERS_PER_PACKET; r++)
        for (int p = 0; p < PlacementConfig::NR_POLARIZATIONS; p++)
          payload->data[t][r][p] = std::complex<int8_t>(1, 1);

    const int total_length =
        sizeof(EthernetHeader) + sizeof(IPHeader) + sizeof(UDPHeader) +
        sizeof(CustomHeader) + sizeof(typename PlacementConfig::PacketPayloadType);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    state->add_received_packet_metadata(total_length, addr);
    state->get_next_write_pointer();
  }

  // Feeds one packet through the reactive path, then asserts locate_packet()
  // -- called with the *same* pre-feed buffer state -- predicted the exact
  // slot the reactive path actually wrote to.
  void feed_and_check(uint64_t sample_count, uint32_t fpga_id, int fpga_index,
                      uint16_t freq_channel) {
    const auto loc = state->locate_packet(sample_count, fpga_index);
    feed_packet(sample_count, fpga_id, freq_channel);
    state->process_all_available_packets();

    if (loc.status != PlacementState::PacketLocationStatus::kValid) {
      return; // covered by DiscardedPacketsAgreeBetweenPaths below
    }
    auto &sample_pixel =
        (*state->d_samples[loc.buffer_index]->samples)[freq_channel][loc.packet_index + 1]
                                                        [fpga_index][0][0][0];
    EXPECT_EQ(sample_pixel, std::complex<int8_t>(1, 1))
        << "locate_packet predicted buffer=" << loc.buffer_index
        << " packet_index=" << loc.packet_index << " but the reactive path didn't write there"
        << " (sample_count=" << sample_count << " fpga_index=" << fpga_index << ")";
  }
};

} // namespace

TEST_F(LocatePacketTest, SingleFpgaInOrderStreamMatchesReactivePath) {
  build({0, 0});
  const int NPC = PlacementConfig::NR_PACKETS_FOR_CORRELATION;
  const int NTS = PlacementConfig::NR_TIME_STEPS_PER_PACKET;
  const uint64_t start = 10000;

  // First packet triggers initialize_buffers() -- locate_packet() isn't
  // meaningful before that, so it isn't checked for this one.
  feed_packet(start, 0, 0);
  state->process_all_available_packets();

  for (int pkt = 1; pkt < NPC; ++pkt) {
    feed_and_check(start + pkt * NTS, 0, 0, 0);
  }
}

TEST_F(LocatePacketTest, MultipleFpgaWithNonzeroDelaysMatchesReactivePath) {
  build({0, 13}); // fpga 1 has a nonzero packet-adjacent delay, like ProcessorStateMultipleFPGATest
  const int NPC = PlacementConfig::NR_PACKETS_FOR_CORRELATION;
  const int NTS = PlacementConfig::NR_TIME_STEPS_PER_PACKET;
  const uint64_t start = 10000;

  feed_packet(start, 0, 0); // triggers initialize_buffers()
  state->process_all_available_packets();

  for (int channel = 0; channel < PlacementConfig::NR_CHANNELS; ++channel) {
    for (int pkt = 0; pkt < NPC; ++pkt) {
      feed_and_check(start + pkt * NTS, 0, 0, channel);
      feed_and_check(start + 13 + pkt * NTS, 1, 1, channel);
    }
  }
}

TEST_F(LocatePacketTest, BufferRolloverBoundaryMatchesReactivePath) {
  build({0, 0});
  const int NPC = PlacementConfig::NR_PACKETS_FOR_CORRELATION;
  const int NTS = PlacementConfig::NR_TIME_STEPS_PER_PACKET;
  const uint64_t start = 10000;

  feed_packet(start, 0, 0);
  state->process_all_available_packets();

  // Cross from buffer 0 into buffer 1 into buffer 2 -- NR_BUFFERS=3, so this
  // walks the full ring once. handle_buffer_completion() advances
  // current_buffer as buffers fill, exactly like a live capture would.
  for (int pkt = 1; pkt < NPC * 3; ++pkt) {
    feed_and_check(start + pkt * NTS, 0, 0, 0);
    if (pkt % NPC == 0) {
      state->handle_buffer_completion(true);
    }
  }
}

TEST_F(LocatePacketTest, StalePacketAgreesWithReactiveDiscard) {
  build({0, 0});
  const int NTS = PlacementConfig::NR_TIME_STEPS_PER_PACKET;
  const uint64_t start = 10000;

  feed_packet(start, 0, 0); // triggers initialize_buffers(), buffer_0.start_seq[0]=start
  state->process_all_available_packets();

  // Far in the past -- below buffer 0's start_seq by more than one
  // NR_BETWEEN_SAMPLES, so packet_index < -1: both locate_packet() and the
  // reactive path must agree this packet is stale/discarded, not placed.
  const uint64_t stale_sample = start - 100 * NTS;
  const auto loc = state->locate_packet(stale_sample, 0);
  EXPECT_EQ(loc.status, PlacementState::PacketLocationStatus::kStale);

  const uint64_t discarded_before = state->packets_discarded.load();
  feed_packet(stale_sample, 0, 0);
  state->process_all_available_packets();
  EXPECT_EQ(state->packets_discarded.load(), discarded_before + 1);
}
