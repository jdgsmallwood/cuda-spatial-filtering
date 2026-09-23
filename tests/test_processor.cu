#include "spatial/common.hpp"
#include "spatial/output.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline_base.hpp"
#include "spatial/spatial.hpp"
#include <array>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

#include <atomic>
#include <chrono>
#include <thread>
#include "support/synthetic_packets.hpp"

// Exercise the real threaded dispatcher, not process_all_available_packets:
// four independent producers, partial batches, delayed commits, ring wrap and
// distinct data at every destination. No window is completed during the feed,
// so its assembled contents can be inspected after all threads have joined.
class FourFpgaHandoffTest : public ::testing::TestWithParam<bool> {};

TEST_P(FourFpgaHandoffTest, PreservesDataAcrossRingWrapAndDelayedCommits) {
  using Cfg = LambdaConfig<4, 4, 64, 40, 2, 10, 256, 1, 64, 32, 1>;
  ProcessorState<Cfg, 3, 256, 3> state(256, 64, 0, {},
                                      {{0, 0}, {1, 1}, {2, 2}, {3, 3}});
  class Sink : public GPUPipeline {
    void execute_pipeline(FinalPacketData *, const bool = false) override {
      ADD_FAILURE() << "Incomplete test window was published";
    }
    void dump_visibilities(const uint64_t = 0) override {}
  } sink;
  state.set_pipeline(&sink);
  state.synchronous_pipeline = true;
  state.use_worker_mailboxes = GetParam();
  constexpr uint64_t start = 640;
  auto sample = [](int fp, int ch, int pkt, int t, int r, int p) {
    return std::complex<int8_t>((fp * 23 + ch * 11 + pkt + t + r + p) % 101,
                                -1 - (pkt + t * 3 + fp + r + p) % 101);
  };
  auto fill = [&](void *dst, int fp, int ch, int pkt) {
    return test_support::build_lambda_wire_packet<Cfg>(
        static_cast<uint8_t *>(dst), start + pkt * 64, fp, ch,
        [&](int t, int r, int p) { return sample(fp, ch, pkt, t, r, p); },
        [&](int r, int p) { return static_cast<int16_t>(1 + fp * 100 + ch * 20 + r * 2 + p); });
  };
  sockaddr_in addr{};
  const int first_len = fill(state.get_current_write_pointer(), 0, 0, 0);
  state.add_received_packet_metadata(first_len, addr);
  state.get_next_write_pointer();
  state.process_all_available_packets();
  state.nr_capture_threads = 4;
  std::thread processor([&] { state.process_packets(); });
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
        const int n = state.reserve_write_batch_strided(fp, linear, target, ptrs, indices);
        for (int i = 0; i < n; ++i)
          lengths[i] = fill(ptrs[i], fp, (flat + i) / 128, (flat + i) % 128);
        if (flat % 17 == 0)
          std::this_thread::sleep_for(std::chrono::microseconds(20));
        state.commit_write_batch(n, indices, lengths, addrs);
        flat += n;
      }
    });
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (state.packets_processed.load() < 2048 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  state.shutdown();
  for (auto &producer : producers) producer.join();
  processor.join();
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

INSTANTIATE_TEST_SUITE_P(LegacyAndMailboxes, FourFpgaHandoffTest,
                        ::testing::Values(false, true));

TEST(FourFpgaWorkerRangeTest, DefersBeyondHorizonWithoutAdvancingWatermarks) {
  using Cfg = LambdaConfig<4, 4, 64, 40, 2, 10, 256, 1, 64, 32, 1>;
  ProcessorState<Cfg, 3, 256, 3> state(256, 64, 0, {},
                                      {{0, 0}, {1, 1}, {2, 2}, {3, 3}});
  auto add = [&](uint64_t seq, int fp) {
    const int length = test_support::build_constant_lambda_wire_packet<Cfg>(
        static_cast<uint8_t *>(state.get_current_write_pointer()), seq, fp, 0,
        {2, -2}, 1);
    state.add_received_packet_metadata(length, {});
    state.get_next_write_pointer();
  };
  add(640, 0);
  ASSERT_EQ(state.process_work_range({0, 1, 1}), 1);
  for (int fp = 0; fp < 4; ++fp)
    add(640 + 4 * 256 * 64, fp);
  EXPECT_EQ(state.process_work_range({1, 5, 1}), 0);
  EXPECT_EQ(state.packets_future_queued.load(), 4);
  EXPECT_EQ(state.packets_stuck_unprocessed.load(), 0);
  for (int fp = 0; fp < 4; ++fp) {
    EXPECT_EQ(state.latest_packet_received[0][fp].load(), fp == 0 ? 640 : 0);
    EXPECT_FALSE(state.d_packet_data[fp + 1]->processed.load());
  }
}

TEST(FourFpgaWorkerRangeTest, DrainsEligibleFuturePacketWhileCaptureIsIdle) {
  using Cfg = LambdaConfig<4, 4, 64, 40, 2, 10, 256, 1, 64, 32, 1>;
  ProcessorState<Cfg, 3, 256, 3> state(256, 64, 0, {},
                                      {{0, 0}, {1, 1}, {2, 2}, {3, 3}});
  class Sink : public GPUPipeline {
    void execute_pipeline(FinalPacketData *, const bool = false) override {}
    void dump_visibilities(const uint64_t = 0) override {}
  } sink;
  state.set_pipeline(&sink);
  state.synchronous_pipeline = true;

  auto add = [&](uint64_t seq) {
    const int length = test_support::build_constant_lambda_wire_packet<Cfg>(
        static_cast<uint8_t *>(state.get_current_write_pointer()), seq, 0, 0,
        {2, -2}, 1);
    state.add_received_packet_metadata(length, {});
    state.get_next_write_pointer();
  };
  constexpr uint64_t start = 640;
  add(start);
  ASSERT_EQ(state.process_work_range({0, 1, 1}), 1);
  add(start + 3 * 256 * 64 + 128);
  ASSERT_EQ(state.process_work_range({1, 2, 1}), 0);
  ASSERT_FALSE(state.d_packet_data[1]->processed.load());

  // A released assembly buffer moves the horizon over the queued packet.
  // No capture thread publishes another ring batch. The processor's idle path
  // must still drain the queued packet so a receiver can reuse its slot.
  state.release_buffer(0);
  state.nr_capture_threads = 4;
  std::thread processor([&] { state.process_packets(); });
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(2);
  while (!state.d_packet_data[1]->processed.load() &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const bool drained = state.d_packet_data[1]->processed.load();
  state.shutdown();
  processor.join();
  EXPECT_TRUE(drained);
  EXPECT_TRUE(state.d_samples[0]->arrivals[0][0][3][0]);
}

TEST(CommonArgsTest, BuildFpgaDelayArrayMapsBySourceIndexUsingFpgaIds) {
  CommonArgs args;
  args.fpga_id_vec = {0, 2, 3};
  args.fpga_ids = {{0, 0}, {2, 1}, {3, 2}};
  args.fpga_delays = {{0, -1000}, {1, 0}, {2, 0}, {3, 1000}};

  const auto delays = build_fpga_delay_array<3>(args);

  EXPECT_EQ(delays[0], -1000);
  EXPECT_EQ(delays[1], 0);
  EXPECT_EQ(delays[2], 1000);
}

// Mock configuration for testing
using TestConfig = LambdaConfig<2,  // NR_CHANNELS
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
using TestMultipleFPGAConfig =
    LambdaConfig<2,  // NR_CHANNELS
                 2,  // NR_FPGA_SOURCES
                 8,  // NR_TIME_STEPS_PER_PACKET
                 20, // NR_RECEIVERS
                 2,  // NR_POLARIZATIONS
                 10, // NR_RECEIVERS_PER_PACKET
                 10, // NR_PACKETS_FOR_CORRELATION
                 1,  // NR_BEAMS
                 32, // NR_PADDED_RECEIVERS
                 32, // NR_PADDED_RECEIVERS_PER_BLOCK
                 1   // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 >;
using TestMultipleFPGAWithOctetConfig =
    LambdaConfig<2,   // NR_CHANNELS
                 4,   // NR_FPGA_SOURCES
                 8,   // NR_TIME_STEPS_PER_PACKET
                 20,  // NR_RECEIVERS
                 2,   // NR_POLARIZATIONS
                 10,  // NR_RECEIVERS_PER_PACKET
                 10,  // NR_PACKETS_FOR_CORRELATION
                 1,   // NR_BEAMS
                 32,  // NR_PADDED_RECEIVERS
                 32,  // NR_PADDED_RECEIVERS_PER_BLOCK
                 1,   // NR_CORRELATED_BLOCKS_TO_ACCUMULATE
                 true // OCTET override
                 >;
constexpr static size_t NR_BUFFERS = 3;
// Simple mock pipeline that just tracks what it receives
class SimpleMockPipeline : public GPUPipeline {
public:
  std::atomic<int> execute_count{0};
  std::atomic<int> release_count{0};
  std::vector<int> buffer_indices_received;
  std::vector<uint64_t> start_seqs_received;
  std::vector<uint64_t> end_seqs_received;
  std::mutex data_mutex;
  FinalPacketData *last_packet_data;

  void execute_pipeline(FinalPacketData *packet_data,
                        const bool dummy_run = false) override {
    {
      std::lock_guard<std::mutex> lock(data_mutex);
      buffer_indices_received.push_back(packet_data->buffer_index);
      start_seqs_received.push_back(packet_data->start_seq_id);
      end_seqs_received.push_back(packet_data->end_seq_id);
      last_packet_data = packet_data;
    }
    execute_count++;

    // Immediately release the buffer to simulate GPU completion
    if (state_) {
      state_->release_buffer(packet_data->buffer_index);
      release_count++;
    }
  }

  void dump_visibilities(const uint64_t end_seq_num = 0) override {}

  int get_execute_count() const { return execute_count.load(); }
  int get_release_count() const { return release_count.load(); }
};

// Test fixture
class ProcessorStateTest : public ::testing::Test {
public:
  ProcessorStateBase *processor_state;
  SimpleMockPipeline *mock_pipeline;

  void SetUp() override {
    std::array<int64_t, 1> delays = {0};
        std::unordered_map<uint32_t, int> map;
        map[0] = 0;
    processor_state = new ProcessorState<TestConfig, NR_BUFFERS>(
        10,                                   // nr_packets_for_correlation
        TestConfig::NR_TIME_STEPS_PER_PACKET, // nr_between_samples
        0,                                    // min_freq_channel
        delays,
        map);

    mock_pipeline = new SimpleMockPipeline();
    mock_pipeline->set_state(processor_state);
    processor_state->set_pipeline(mock_pipeline);
    processor_state->synchronous_pipeline = true;
  }

  void TearDown() override {
    processor_state->running = 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    delete processor_state;
    delete mock_pipeline;
  }

  // Helper to create a properly formatted Lambda packet
  virtual void create_lambda_packet(uint64_t sample_count, uint32_t fpga_id,
                                    uint16_t freq_channel, int val,
                                    int /*desired_src_ip_third_octet*/ = 0) {
    // Get the current write pointer
    void *write_ptr = processor_state->get_current_write_pointer();
    uint8_t *data_ptr = (uint8_t *)write_ptr;

    // Ethernet header (14 bytes)
    EthernetHeader *eth = (EthernetHeader *)data_ptr;
    memset(eth, 0, sizeof(EthernetHeader));
    eth->ethertype = htons(0x0800); // IPv4
    data_ptr += sizeof(EthernetHeader);

    // IP header (20 bytes)
    IPHeader *ip = (IPHeader *)data_ptr;
    memset(ip, 0, sizeof(IPHeader));
    ip->version_ihl = 0x45; // IPv4, 20 byte header
    data_ptr += sizeof(IPHeader);

    // UDP header (8 bytes)
    UDPHeader *udp = (UDPHeader *)data_ptr;
    memset(udp, 0, sizeof(UDPHeader));
    data_ptr += sizeof(UDPHeader);

    // Custom header (22 bytes: 8 + 4 + 2 + 8)
    CustomHeader *custom = (CustomHeader *)data_ptr;
    custom->sample_count = sample_count;
    custom->fpga_id = fpga_id;
    custom->freq_channel = freq_channel;
    memset(custom->padding, 0, sizeof(custom->padding));
    data_ptr += sizeof(CustomHeader);

    // Payload: PacketPayload<PacketScaleStructure, PacketDataStructure>
    // PacketScaleStructure: int16_t[NR_RECEIVERS][NR_POLARIZATIONS]
    // PacketDataStructure:
    // complex<int8_t>[NR_TIME_STEPS][NR_RECEIVERS][NR_POLARIZATIONS]
    auto *payload =
        reinterpret_cast<typename TestConfig::PacketPayloadType *>(data_ptr);

    // Fill scales with test data
    for (int r = 0; r < TestConfig::NR_RECEIVERS_PER_PACKET; r++) {
      for (int p = 0; p < TestConfig::NR_POLARIZATIONS; p++) {
        payload->scales[r][p] = static_cast<int16_t>(1 + r); // Non-zero scales
      }
    }

    // Fill data with test pattern
    for (int t = 0; t < TestConfig::NR_TIME_STEPS_PER_PACKET; t++) {
      for (int r = 0; r < TestConfig::NR_RECEIVERS_PER_PACKET; r++) {
        for (int p = 0; p < TestConfig::NR_POLARIZATIONS; p++) {
          payload->data[t][r][p] = std::complex<int8_t>(
              static_cast<int8_t>(val), static_cast<int8_t>(val));
        }
      }
    }

    // Set packet metadata
    int total_length = sizeof(EthernetHeader) + sizeof(IPHeader) +
                       sizeof(UDPHeader) + sizeof(CustomHeader) +
                       sizeof(typename TestConfig::PacketPayloadType);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    processor_state->add_received_packet_metadata(total_length, addr);
  }

  void validate_packet_contents(FinalPacketData *parcel, int channel,
                                int packet_number, int val) {
    typename TestConfig::InputPacketSamplesType *packet_samples =
        (typename TestConfig::InputPacketSamplesType *)
            parcel->get_samples_ptr();
    for (int t = 0; t < TestConfig::NR_TIME_STEPS_PER_PACKET; t++) {
      for (int r = 0; r < TestConfig::NR_RECEIVERS_PER_PACKET; r++) {
        for (int p = 0; p < TestConfig::NR_POLARIZATIONS; p++) {
          EXPECT_EQ(packet_samples[0][channel][packet_number + 1][0][t][r]
                                  [p], // [0] is for FPGA and there is only one
                                       // in the config.
                    std::complex<int8_t>(static_cast<int8_t>(val),
                                         static_cast<int8_t>(val)));
        }
      }
    }
  }

  // Add packet and advance write pointer
  virtual void add_packet(uint64_t sample_count, uint32_t fpga_id,
                          uint16_t freq_channel, int val = 1,
                          int desired_src_ip_third_octet = 0) {
    create_lambda_packet(sample_count, fpga_id, freq_channel, val,
                         desired_src_ip_third_octet);
    processor_state->get_next_write_pointer();
  }
};

class ProcessorStateMultipleFPGATest : public ProcessorStateTest {

  void SetUp() override {
    std::array<int64_t, TestMultipleFPGAConfig::NR_FPGA_SOURCES> delays = {0,
                                                                           13};
        std::unordered_map<uint32_t, int> map;
        map[0] = 0;
        map[1] = 1;
    processor_state = new ProcessorState<TestMultipleFPGAConfig, NR_BUFFERS>(
        10, // nr_packets_for_correlation
        TestMultipleFPGAConfig::NR_TIME_STEPS_PER_PACKET, // nr_between_samples
        0,                                                // min_freq_channel
        delays, map);

    mock_pipeline = new SimpleMockPipeline();
    mock_pipeline->set_state(processor_state);
    processor_state->set_pipeline(mock_pipeline);
    processor_state->synchronous_pipeline = true;
  }

public:
  void create_lambda_packet(uint64_t sample_count, uint32_t fpga_id,
                            uint16_t freq_channel, int val,
                            int /*desired_src_ip_third_octet*/ = 0) override {
    // Get the current write pointer
    void *write_ptr = processor_state->get_current_write_pointer();
    uint8_t *data_ptr = (uint8_t *)write_ptr;

    // Ethernet header (14 bytes)
    EthernetHeader *eth = (EthernetHeader *)data_ptr;
    memset(eth, 0, sizeof(EthernetHeader));
    eth->ethertype = htons(0x0800); // IPv4
    data_ptr += sizeof(EthernetHeader);

    // IP header (20 bytes)
    IPHeader *ip = (IPHeader *)data_ptr;
    memset(ip, 0, sizeof(IPHeader));
    ip->version_ihl = 0x45; // IPv4, 20 byte header
    data_ptr += sizeof(IPHeader);

    // UDP header (8 bytes)
    UDPHeader *udp = (UDPHeader *)data_ptr;
    memset(udp, 0, sizeof(UDPHeader));
    data_ptr += sizeof(UDPHeader);

    // Custom header (22 bytes: 8 + 4 + 2 + 8)
    CustomHeader *custom = (CustomHeader *)data_ptr;
    custom->sample_count = sample_count;
    custom->fpga_id = fpga_id;
    custom->freq_channel = freq_channel;
    memset(custom->padding, 0, sizeof(custom->padding));
    data_ptr += sizeof(CustomHeader);

    // Payload: PacketPayload<PacketScaleStructure, PacketDataStructure>
    // PacketScaleStructure: int16_t[NR_RECEIVERS][NR_POLARIZATIONS]
    // PacketDataStructure:
    // complex<int8_t>[NR_TIME_STEPS][NR_RECEIVERS][NR_POLARIZATIONS]
    auto *payload =
        reinterpret_cast<typename TestMultipleFPGAConfig::PacketPayloadType *>(
            data_ptr);

    // Fill scales with test data
    for (int r = 0; r < TestMultipleFPGAConfig::NR_RECEIVERS_PER_PACKET; r++) {
      for (int p = 0; p < TestMultipleFPGAConfig::NR_POLARIZATIONS; p++) {
        payload->scales[r][p] =
            static_cast<int16_t>(1 + r + fpga_id); // Non-zero scales
      }
    }

    // Fill data with test pattern
    for (int t = 0; t < TestMultipleFPGAConfig::NR_TIME_STEPS_PER_PACKET; t++) {
      for (int r = 0; r < TestMultipleFPGAConfig::NR_RECEIVERS_PER_PACKET;
           r++) {
        for (int p = 0; p < TestMultipleFPGAConfig::NR_POLARIZATIONS; p++) {
          payload->data[t][r][p] = std::complex<int8_t>(
              static_cast<int8_t>(val), static_cast<int8_t>(val));
        }
      }
    }

    // Set packet metadata
    int total_length =
        sizeof(EthernetHeader) + sizeof(IPHeader) + sizeof(UDPHeader) +
        sizeof(CustomHeader) +
        sizeof(typename TestMultipleFPGAConfig::PacketPayloadType);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    constexpr int desired_src_ip_third_octet = 0;
    uint32_t host_ip =
        0x0a000001; // This is (10 << 24) | (0 << 16) | (0 << 8) | 1

    // 2. Clear the current 3rd octet (bits 8-15) and set the new one
    // The mask ~0x0000FF00U clears the 3rd octet.
    host_ip &= ~0x0000FF00U;

    // 3. Set the new 3rd octet value
    host_ip |= (static_cast<uint32_t>(desired_src_ip_third_octet) << 8);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(host_ip);

    processor_state->add_received_packet_metadata(total_length, addr);
  }
};

class ProcessorStateMultipleFPGAWithOctetTest
    : public ProcessorStateMultipleFPGATest {

  void SetUp() override {
    using MapType = std::unordered_map<uint32_t, int>;
 MapType fpga_ids{
    {10, 0},
    {11, 1},
    {12, 2},
    {13, 3}
}; 

    std::array<int64_t, TestMultipleFPGAWithOctetConfig::NR_FPGA_SOURCES>
        delays = {0, 13, 26, 4};
    processor_state = new ProcessorState<TestMultipleFPGAWithOctetConfig,
                                         NR_BUFFERS>(
        10, // nr_packets_for_correlation
        TestMultipleFPGAConfig::NR_TIME_STEPS_PER_PACKET, // nr_between_samples
        0,                                                // min_freq_channel
        delays, fpga_ids);

    mock_pipeline = new SimpleMockPipeline();
    mock_pipeline->set_state(processor_state);
    processor_state->set_pipeline(mock_pipeline);
    processor_state->synchronous_pipeline = true;
  }

public:
  void create_lambda_packet(uint64_t sample_count, uint32_t fpga_id,
                            uint16_t freq_channel, int val,
                            int desired_src_ip_third_octet = 0) override {
    // Get the current write pointer
    void *write_ptr = processor_state->get_current_write_pointer();
    uint8_t *data_ptr = (uint8_t *)write_ptr;

    // Ethernet header (14 bytes)
    EthernetHeader *eth = (EthernetHeader *)data_ptr;
    memset(eth, 0, sizeof(EthernetHeader));
    eth->ethertype = htons(0x0800); // IPv4
    data_ptr += sizeof(EthernetHeader);

    // IP header (20 bytes)
    IPHeader *ip = (IPHeader *)data_ptr;
    memset(ip, 0, sizeof(IPHeader));
    ip->version_ihl = 0x45; // IPv4, 20 byte header
    data_ptr += sizeof(IPHeader);

    // UDP header (8 bytes)
    UDPHeader *udp = (UDPHeader *)data_ptr;
    memset(udp, 0, sizeof(UDPHeader));
    data_ptr += sizeof(UDPHeader);

    // Custom header (22 bytes: 8 + 4 + 2 + 8)
    CustomHeader *custom = (CustomHeader *)data_ptr;
    custom->sample_count = sample_count;
    custom->fpga_id = fpga_id;
    custom->freq_channel = freq_channel;
    memset(custom->padding, 0, sizeof(custom->padding));
    data_ptr += sizeof(CustomHeader);

    // Payload: PacketPayload<PacketScaleStructure, PacketDataStructure>
    // PacketScaleStructure: int16_t[NR_RECEIVERS][NR_POLARIZATIONS]
    // PacketDataStructure:
    // complex<int8_t>[NR_TIME_STEPS][NR_RECEIVERS][NR_POLARIZATIONS]
    auto *payload =
        reinterpret_cast<typename TestMultipleFPGAConfig::PacketPayloadType *>(
            data_ptr);

    // Fill scales with test data
    for (int r = 0; r < TestMultipleFPGAConfig::NR_RECEIVERS_PER_PACKET; r++) {
      for (int p = 0; p < TestMultipleFPGAConfig::NR_POLARIZATIONS; p++) {
        payload->scales[r][p] =
            static_cast<int16_t>(1 + r + fpga_id); // Non-zero scales
      }
    }

    // Fill data with test pattern
    for (int t = 0; t < TestMultipleFPGAConfig::NR_TIME_STEPS_PER_PACKET; t++) {
      for (int r = 0; r < TestMultipleFPGAConfig::NR_RECEIVERS_PER_PACKET;
           r++) {
        for (int p = 0; p < TestMultipleFPGAConfig::NR_POLARIZATIONS; p++) {
          payload->data[t][r][p] = std::complex<int8_t>(
              static_cast<int8_t>(val), static_cast<int8_t>(val));
        }
      }
    }

    // Set packet metadata
    int total_length =
        sizeof(EthernetHeader) + sizeof(IPHeader) + sizeof(UDPHeader) +
        sizeof(CustomHeader) +
        sizeof(typename TestMultipleFPGAConfig::PacketPayloadType);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));

    uint32_t host_ip =
        0x0a000001; // This is (10 << 24) | (0 << 16) | (0 << 8) | 1

    // 2. Clear the current 3rd octet (bits 8-15) and set the new one
    // The mask ~0x0000FF00U clears the 3rd octet.
    host_ip &= ~0x0000FF00U;

    // 3. Set the new 3rd octet value
    host_ip |= (static_cast<uint32_t>(desired_src_ip_third_octet) << 8);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(host_ip);

    processor_state->add_received_packet_metadata(total_length, addr);
  }

  void add_packet(uint64_t sample_count, uint32_t fpga_id,
                  uint16_t freq_channel, int val = 1,
                  int desired_src_ip_third_octet = 0) override {
    create_lambda_packet(sample_count, fpga_id, freq_channel, val,
                         desired_src_ip_third_octet);
    processor_state->get_next_write_pointer();
  }
};

class ProcessorStateHugeDelayBootstrapTest
    : public ProcessorStateMultipleFPGAWithOctetTest {
  void SetUp() override {
    using MapType = std::unordered_map<uint32_t, int>;
    MapType fpga_ids{{10, 0}, {11, 1}, {12, 2}, {13, 3}};
    std::array<int64_t, TestMultipleFPGAWithOctetConfig::NR_FPGA_SOURCES>
        delays = {8'800'000, 0, -80'800'000'000, -16'800'000};
    processor_state =
        new ProcessorState<TestMultipleFPGAWithOctetConfig, NR_BUFFERS>(
            10, TestMultipleFPGAWithOctetConfig::NR_TIME_STEPS_PER_PACKET,
            0, delays, fpga_ids);

    mock_pipeline = new SimpleMockPipeline();
    mock_pipeline->set_state(processor_state);
    processor_state->set_pipeline(mock_pipeline);
    processor_state->synchronous_pipeline = true;
  }
};

TEST_F(ProcessorStateHugeDelayBootstrapTest,
       FirstPacketInitializesOnceAndLaterStreamsConverge) {
  constexpr uint64_t canonical_start = 100'000'000'000ULL;
  constexpr std::array<int64_t, 4> delays = {
      8'800'000, 0, -80'800'000'000, -16'800'000};
  // Source 0 arrives first but is 24 samples later in canonical time than
  // source 2. It remains the one and only reference; the two packets more
  // than one overlap slot earlier may be discarded, but must not re-seed.
  constexpr std::array<uint64_t, 4> canonical_offsets = {24, 16, 0, 8};

  for (int source = 0; source < 4; ++source) {
    const uint64_t sample_count = static_cast<uint64_t>(
        static_cast<int64_t>(canonical_start + canonical_offsets[source]) +
        delays[source]);
    add_packet(sample_count, source, 0, source + 1, source + 10);
  }

  processor_state->process_all_available_packets();

  // packets_processed is the number of ring packets handled, including the
  // two intentionally discarded startup packets.
  EXPECT_EQ(processor_state->packets_processed.load(), 4);
  EXPECT_EQ(processor_state->packets_discarded.load(), 2);
  EXPECT_EQ(processor_state->packets_stuck_unprocessed.load(), 0);

  auto *state = static_cast<ProcessorState<
      TestMultipleFPGAWithOctetConfig, NR_BUFFERS> *>(processor_state);
  for (int source = 0; source < 4; ++source) {
    EXPECT_EQ(state->buffers[0].start_seq[source],
              static_cast<uint64_t>(
                  static_cast<int64_t>(canonical_start + 24) + delays[source]));
  }

  // Once every source has advanced beyond the selected startup reference,
  // all four place normally without changing the established boundaries.
  for (int source = 0; source < 4; ++source) {
    const uint64_t sample_count = static_cast<uint64_t>(
        static_cast<int64_t>(canonical_start + 32) + delays[source]);
    add_packet(sample_count, source, 0, source + 1, source + 10);
  }
  processor_state->process_all_available_packets();

  EXPECT_EQ(processor_state->packets_processed.load(), 8);
  EXPECT_EQ(processor_state->packets_discarded.load(), 2);
  EXPECT_EQ(processor_state->packets_stuck_unprocessed.load(), 0);
}

TEST_F(ProcessorStateTest, ProcessSinglePacketTest) {
  add_packet(1000, 0, 0);

  processor_state->process_all_available_packets();

  EXPECT_EQ(processor_state->packets_processed, 1);
}

TEST_F(ProcessorStateTest, FillOneBufferTest) {
  const uint64_t start_sample = 1000;

  // For one complete buffer, we need packets from all channels and all FPGAs
  // Total packets = NR_CHANNELS * NR_FPGA_SOURCES *
  // NR_PACKETS_FOR_CORRELATION
  int total_packets = TestConfig::NR_CHANNELS * TestConfig::NR_FPGA_SOURCES *
                      (TestConfig::NR_PACKETS_FOR_CORRELATION + 2);

  for (int channel = 0; channel < TestConfig::NR_CHANNELS; channel++) {
    for (int fpga = 0; fpga < TestConfig::NR_FPGA_SOURCES; fpga++) {
      for (int pkt = 0; pkt < TestConfig::NR_PACKETS_FOR_CORRELATION + 1;
           pkt++) {
        uint64_t sample =
            start_sample + pkt * TestConfig::NR_TIME_STEPS_PER_PACKET;
        add_packet(sample, fpga, channel);
      }

      // do this afterwards so it actually is right. Sample num initialization
      // is done off the first packet received.
      for (int pkt = -1; pkt < 0; pkt++) {
        uint64_t sample =
            start_sample + pkt * TestConfig::NR_TIME_STEPS_PER_PACKET;
        add_packet(sample, fpga, channel);
      }
    }
  }

  processor_state->process_all_available_packets();
  processor_state->handle_buffer_completion(true);
  EXPECT_EQ(processor_state->packets_processed, total_packets);
  EXPECT_EQ(mock_pipeline->get_execute_count(), 1);

  validate_packet_contents(mock_pipeline->last_packet_data, 0, 0, 1);
}

TEST_F(ProcessorStateTest, OnePacketPlacementTest) {
  const uint64_t start_sample = 1000;
  // need to initialize - so give it a start_sample
  int pkt = 1;
  int fpga = 0;
  int channel = 0;
  uint64_t sample = start_sample + pkt * TestConfig::NR_TIME_STEPS_PER_PACKET;
  add_packet(start_sample, fpga, channel, 1);
  add_packet(sample, fpga, channel, 10);

  processor_state->process_all_available_packets();
  processor_state->handle_buffer_completion(true);
  EXPECT_EQ(processor_state->packets_processed, 2);
  EXPECT_EQ(mock_pipeline->get_execute_count(), 1);

  validate_packet_contents(mock_pipeline->last_packet_data, 0, 0, 1);
  validate_packet_contents(mock_pipeline->last_packet_data, channel, pkt, 10);
}

TEST_F(ProcessorStateTest, OnePacketPlacementTest2) {
  const uint64_t start_sample = 1000;
  // need to initialize - so give it a start_sample
  int pkt = 2;
  int fpga = 0;
  int channel = 1;
  uint64_t sample = start_sample + pkt * TestConfig::NR_TIME_STEPS_PER_PACKET;
  add_packet(start_sample, fpga, channel, 1);
  add_packet(sample, fpga, channel, 10);

  processor_state->process_all_available_packets();
  processor_state->handle_buffer_completion(true);
  EXPECT_EQ(processor_state->packets_processed, 2);
  EXPECT_EQ(mock_pipeline->get_execute_count(), 1);

  validate_packet_contents(mock_pipeline->last_packet_data, 1, 0, 1);
  validate_packet_contents(mock_pipeline->last_packet_data, channel, pkt, 10);
}

TEST_F(ProcessorStateTest, DiscardOldPacketsTest) {
  // Initialize with a packet at sample 1000
  add_packet(1000, 0, 0);
  processor_state->process_all_available_packets();

  // Try to add a packet that's too old (before current buffer)
  add_packet(500, 0, 0);
  processor_state->process_all_available_packets();
  // This packet should be discarded
  EXPECT_GT(processor_state->packets_discarded, 0);
}

TEST_F(ProcessorStateTest, MissingPacketHandlingTest) {
  // fill up buffers so that there is definitely a non-zero scale in
  // every slot.
  int start_sample = 1000;
  for (int buf = 0; buf < NR_BUFFERS; ++buf) {
    for (int channel = 0; channel < TestConfig::NR_CHANNELS; channel++) {
      for (int fpga = 0; fpga < TestConfig::NR_FPGA_SOURCES; fpga++) {
        for (int pkt = 0; pkt < TestConfig::NR_PACKETS_FOR_CORRELATION; pkt++) {
          uint64_t sample = start_sample +
                            buf * TestConfig::NR_PACKETS_FOR_CORRELATION *
                                TestConfig::NR_TIME_STEPS_PER_PACKET +
                            pkt * TestConfig::NR_TIME_STEPS_PER_PACKET;
          add_packet(sample, fpga, channel);
        }
      }
    }
  }
  processor_state->process_all_available_packets();
  processor_state->handle_buffer_completion();

  INFO_LOG("Finished initial.");
  // add two packets that are way further along (beyond the buffer horizon),
  // this will cause the pipeline to run w/ missing packets.  Beyond-horizon
  // packets no longer advance the completion watermark at parse time (that
  // raced concurrent capture -- see copy_data_to_input_buffer_if_able);
  // recovery now runs through the drain stall safety net, which the tests
  // trigger explicitly with a zero threshold instead of waiting out the
  // production hysteresis (default 100 ms).
  add_packet(20000, 0, 0);
  add_packet(20000, 0, 1);
  INFO_LOG("Packets added");
  processor_state->process_all_available_packets();
  auto *concrete_state =
      static_cast<ProcessorState<TestConfig, NR_BUFFERS> *>(processor_state);
  concrete_state->future_stall_force_threshold = std::chrono::milliseconds(0);
  std::array<uint64_t, TestConfig::NR_FPGA_SOURCES> gmax{};
  concrete_state->get_global_max_packet_array(gmax);
  concrete_state->drain_future_packets(gmax); // arms the stall timer
  concrete_state->drain_future_packets(gmax); // fires the safety net
  processor_state->handle_buffer_completion();
  INFO_LOG("Firing again...");
  EXPECT_EQ(processor_state->packets_missing, 2 * 20);

  int16_t *scales_last_packet =
      (int16_t *)mock_pipeline->last_packet_data->get_scales_ptr();
  int scales_length =
      mock_pipeline->last_packet_data->get_scales_element_size() /
      sizeof(int16_t);

  bool *arrivals_last_packet =
      (bool *)mock_pipeline->last_packet_data->get_arrivals_ptr();
  int arrivals_length =
      mock_pipeline->last_packet_data->get_arrivals_size() / sizeof(bool);
  // If missing - all scales should be zero.
  for (int i = 0; i < scales_length; ++i) {
    EXPECT_EQ(scales_last_packet[i], 0);
  }
}

// Regression test for the completion-watermark race: a packet deferred to
// future_packet_queue (beyond the buffer horizon) must NOT advance
// latest_packet_received at parse time.  The old parse-time bump let a
// buffer complete off a deferred packet's sample count while the buffer
// still had holes that in-flight packets were about to fill -- under
// concurrent capture threads (nr_capture_threads > 0) or bounded stream
// skew this silently zero-filled real data as "missing".
TEST_F(ProcessorStateTest, FuturePacketDoesNotPrematurelyCompleteBuffer) {
  const uint64_t start_sample = 1000;
  constexpr int NTS = TestConfig::NR_TIME_STEPS_PER_PACKET;
  constexpr int NPC = TestConfig::NR_PACKETS_FOR_CORRELATION;

  // Channel 1: complete window (pkt 0..NPC then the -1 front packet),
  // mirroring MissingPacketCountIsZeroForCompleteBuffer's fill.
  for (int pkt = 0; pkt < NPC + 1; pkt++) {
    add_packet(start_sample + pkt * NTS, 0, 1);
  }
  add_packet(start_sample - NTS, 0, 1);

  // Channel 0: only the head of the window (pkt 0..4) -- the tail is still
  // "in flight", so channel 0's watermark must hold the buffer open.
  for (int pkt = 0; pkt <= 4; pkt++) {
    add_packet(start_sample + pkt * NTS, 0, 0);
  }
  processor_state->process_all_available_packets();

  // A far-future channel-0 packet (beyond the last buffer's window) gets
  // deferred to future_packet_queue.  It must not complete the buffer.
  add_packet(start_sample + 500 * NTS, 0, 0);
  processor_state->process_all_available_packets();
  processor_state->handle_buffer_completion();

  EXPECT_EQ(mock_pipeline->get_execute_count(), 0);
  EXPECT_EQ(processor_state->packets_missing, 0);

  // The in-flight tail (pkt 5..NPC and the -1 front packet) arrives; now the
  // buffer completes with nothing missing.
  for (int pkt = 5; pkt < NPC + 1; pkt++) {
    add_packet(start_sample + pkt * NTS, 0, 0);
  }
  add_packet(start_sample - NTS, 0, 0);
  processor_state->process_all_available_packets();
  processor_state->handle_buffer_completion();

  EXPECT_EQ(mock_pipeline->get_execute_count(), 1);
  EXPECT_EQ(processor_state->packets_missing, 0);
}

TEST_F(ProcessorStateMultipleFPGATest, MultipleFPGABasicTest) {

  int start_sample = 1000;
  for (int channel = 0; channel < TestMultipleFPGAConfig::NR_CHANNELS;
       channel++) {
    for (int fpga = 0; fpga < TestMultipleFPGAConfig::NR_FPGA_SOURCES; fpga++) {
      for (int pkt = 0;
           pkt < TestMultipleFPGAConfig::NR_PACKETS_FOR_CORRELATION; pkt++) {
        uint64_t sample =
            start_sample +
            pkt * TestMultipleFPGAConfig::NR_TIME_STEPS_PER_PACKET;
        add_packet(sample, fpga, channel);
      }
    }

    processor_state->process_all_available_packets();
    processor_state->handle_buffer_completion();
  }

  EXPECT_EQ(processor_state->packets_missing, 0);
}

TEST_F(ProcessorStateMultipleFPGATest, MultipleFPGAPlacementTest) {
  // This will give different values to different FPGAs.
  // We'll use 1 and 2 so that the second batch of receivers should
  // be double the first.

  int start_sample = 1000;
  for (int channel = 0; channel < TestMultipleFPGAConfig::NR_CHANNELS;
       channel++) {
    for (int fpga = 0; fpga < TestMultipleFPGAConfig::NR_FPGA_SOURCES; fpga++) {
      for (int pkt = 0;
           pkt < TestMultipleFPGAConfig::NR_PACKETS_FOR_CORRELATION; pkt++) {
        uint64_t sample =
            start_sample +
            pkt * TestMultipleFPGAConfig::NR_TIME_STEPS_PER_PACKET;
        add_packet(sample, fpga, channel, fpga + 1);
      }
    }
  }

  processor_state->process_all_available_packets();
  processor_state->handle_buffer_completion(true);
  EXPECT_EQ(processor_state->packets_missing, 4);

  typename TestMultipleFPGAConfig::InputPacketSamplesType *samples =
      (typename TestMultipleFPGAConfig::InputPacketSamplesType *)
          mock_pipeline->last_packet_data->get_samples_ptr();

  for (int channel = 0; channel < TestMultipleFPGAConfig::NR_CHANNELS;
       channel++) {
    for (int receiver = 0;
         receiver < TestMultipleFPGAConfig::NR_RECEIVERS_PER_PACKET;
         receiver++) {
      for (int pkt = -1;
           pkt < static_cast<int>(
                     TestMultipleFPGAConfig::NR_PACKETS_FOR_CORRELATION) +
                     1;
           pkt++) {
        for (int fpga = 0; fpga < TestMultipleFPGAConfig::NR_FPGA_SOURCES;
             fpga++) {
          for (int t = 0; t < TestMultipleFPGAConfig::NR_TIME_STEPS_PER_PACKET;
               t++) {
            for (int pol = 0; pol < TestMultipleFPGAConfig::NR_POLARIZATIONS;
                 pol++) {
              std::complex<int8_t> expected_value;
              if ((fpga == 0 &&
                   (pkt == -1 ||
                    pkt == static_cast<int>(TestMultipleFPGAConfig::
                                                NR_PACKETS_FOR_CORRELATION))) ||
                  (fpga == 1 &&
                   pkt >=
                       static_cast<int>(
                           TestMultipleFPGAConfig::NR_PACKETS_FOR_CORRELATION) -
                           2)) {
                expected_value = {static_cast<int8_t>(0),
                                  static_cast<int8_t>(0)};
              } else {
                expected_value = {static_cast<int8_t>(fpga + 1),
                                  static_cast<int8_t>(fpga + 1)};
              }
              EXPECT_EQ(samples[0][channel][pkt + 1][fpga][t][receiver][pol],
                        expected_value)
                  << "Mismatch at channel " << channel << " pkt " << pkt
                  << " fpga " << fpga << " t " << t << " receiver " << receiver
                  << " pol" << pol << std::endl;
            }
          }
        }
      }
    }
  }
}

TEST_F(ProcessorStateTest, MissingPacketCountIsZeroForCompleteBuffer) {
  // A fully-received buffer (all channels and FPGAs present) must not
  // increment packets_missing.  release_buffer() resets the arrivals array
  // immediately when the mock pipeline calls it, so we check the accumulated
  // counter on the ProcessorState rather than calling get_num_missing_packets()
  // directly on last_packet_data after execution.
  const uint64_t start_sample = 1000;

  for (int channel = 0; channel < TestConfig::NR_CHANNELS; channel++) {
    for (int fpga = 0; fpga < TestConfig::NR_FPGA_SOURCES; fpga++) {
      for (int pkt = 0; pkt < TestConfig::NR_PACKETS_FOR_CORRELATION + 1;
           pkt++) {
        uint64_t sample =
            start_sample + pkt * TestConfig::NR_TIME_STEPS_PER_PACKET;
        add_packet(sample, fpga, channel);
      }
      for (int pkt = -1; pkt < 0; pkt++) {
        uint64_t sample =
            start_sample + pkt * TestConfig::NR_TIME_STEPS_PER_PACKET;
        add_packet(sample, fpga, channel);
      }
    }
  }

  processor_state->process_all_available_packets();
  processor_state->handle_buffer_completion(true);

  EXPECT_EQ(mock_pipeline->get_execute_count(), 1);
  EXPECT_EQ(processor_state->packets_missing, 0);
}

TEST_F(ProcessorStateTest, SequenceNumbersAreSetFromFirstFPGA) {
  // start_seq_id and end_seq_id on the FinalPacketData passed to the pipeline
  // must be the FPGA-0 window boundaries for the completed buffer.
  // With start_sample=1000, NR_PACKETS_FOR_CORRELATION=10, NR_TIME_STEPS=8
  // and no FPGA delays:
  //   start_seq_id = 1000
  //   end_seq_id   = 1000 + (10-1)*8 = 1072
  const uint64_t start_sample = 1000;
  const uint64_t expected_start =
      start_sample; // buffer[0].start_seq[0] after initialize_buffers
  const uint64_t expected_end =
      start_sample +
      (TestConfig::NR_PACKETS_FOR_CORRELATION - 1) *
          TestConfig::NR_TIME_STEPS_PER_PACKET; // 1072

  for (int channel = 0; channel < TestConfig::NR_CHANNELS; channel++) {
    for (int fpga = 0; fpga < TestConfig::NR_FPGA_SOURCES; fpga++) {
      for (int pkt = 0; pkt < TestConfig::NR_PACKETS_FOR_CORRELATION + 1;
           pkt++) {
        uint64_t sample =
            start_sample + pkt * TestConfig::NR_TIME_STEPS_PER_PACKET;
        add_packet(sample, fpga, channel);
      }
      for (int pkt = -1; pkt < 0; pkt++) {
        uint64_t sample =
            start_sample + pkt * TestConfig::NR_TIME_STEPS_PER_PACKET;
        add_packet(sample, fpga, channel);
      }
    }
  }

  processor_state->process_all_available_packets();
  processor_state->handle_buffer_completion(true);

  ASSERT_EQ(mock_pipeline->get_execute_count(), 1);
  EXPECT_EQ(mock_pipeline->start_seqs_received[0], expected_start);
  EXPECT_EQ(mock_pipeline->end_seqs_received[0], expected_end);
}

// Helper: builds and submits one TestConfig-shaped wire packet directly into
// an arbitrary ProcessorStateBase (not the fixture's processor_state).
static void add_packet_to_state(ProcessorStateBase *state,
                                uint64_t sample_count, uint32_t fpga_id,
                                uint16_t freq_channel, int val = 1) {
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
      reinterpret_cast<typename TestConfig::PacketPayloadType *>(data_ptr);
  for (int r = 0; r < TestConfig::NR_RECEIVERS_PER_PACKET; r++)
    for (int p = 0; p < TestConfig::NR_POLARIZATIONS; p++)
      payload->scales[r][p] = static_cast<int16_t>(1);
  for (int t = 0; t < TestConfig::NR_TIME_STEPS_PER_PACKET; t++)
    for (int r = 0; r < TestConfig::NR_RECEIVERS_PER_PACKET; r++)
      for (int p = 0; p < TestConfig::NR_POLARIZATIONS; p++)
        payload->data[t][r][p] =
            std::complex<int8_t>(static_cast<int8_t>(val), static_cast<int8_t>(val));

  const int total_length =
      sizeof(EthernetHeader) + sizeof(IPHeader) + sizeof(UDPHeader) +
      sizeof(CustomHeader) + sizeof(typename TestConfig::PacketPayloadType);

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  state->add_received_packet_metadata(total_length, addr);
  state->get_next_write_pointer();
}

TEST(ProcessorStateChannelFilterTest, PacketsOutsideFreqRangeAreDiscarded) {
  // With min_freq_channel=3 and NR_CHANNELS=2, only channels 3 (slot 0)
  // and 4 (slot 1) are valid. Channels 2 (below min) and 5 (above max)
  // must be discarded without incrementing packets_processed.
  std::array<int64_t, TestConfig::NR_FPGA_SOURCES> delays = {0};
  std::unordered_map<uint32_t, int> fpga_map;
  fpga_map[0] = 0;

  auto *state = new ProcessorState<TestConfig, NR_BUFFERS>(
      TestConfig::NR_PACKETS_FOR_CORRELATION,
      TestConfig::NR_TIME_STEPS_PER_PACKET,
      3, // min_freq_channel
      delays, fpga_map);

  SimpleMockPipeline pipeline;
  pipeline.set_state(state);
  state->set_pipeline(&pipeline);
  state->synchronous_pipeline = true;

  // channel 2: 2 - 3 = -1 < 0 → discarded
  add_packet_to_state(state, 1000, 0, 2);
  // channel 5: 5 - 3 = 2 >= NR_CHANNELS=2 → discarded
  add_packet_to_state(state, 1000, 0, 5);
  // channel 3: 3 - 3 = 0, slot 0 → accepted
  add_packet_to_state(state, 1000, 0, 3);
  // channel 4: 4 - 3 = 1, slot 1 → accepted
  add_packet_to_state(state, 1000, 0, 4);

  state->process_all_available_packets();

  EXPECT_EQ(state->packets_discarded, 2);
  EXPECT_EQ(state->packets_processed, 2);

  state->running = 0;
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  delete state;
}

TEST_F(ProcessorStateMultipleFPGAWithOctetTest,
       MultipleFPGAPlacementWithDifferentIDTest) {
  // This will give different values to different FPGAs.
  // We'll use 1 and 2 so that the second batch of receivers should
  // be double the first.

  int start_sample = 1000;
  for (int channel = 0; channel < TestMultipleFPGAWithOctetConfig::NR_CHANNELS;
       channel++) {
    for (int fpga = 0; fpga < TestMultipleFPGAWithOctetConfig::NR_FPGA_SOURCES;
         fpga++) {
      for (int pkt = 0;
           pkt < TestMultipleFPGAWithOctetConfig::NR_PACKETS_FOR_CORRELATION;
           pkt++) {
        uint64_t sample =
            start_sample +
            pkt * TestMultipleFPGAWithOctetConfig::NR_TIME_STEPS_PER_PACKET;
        add_packet(sample, fpga, channel, fpga + 1, fpga + 10);
      }
    }
  }

  processor_state->process_all_available_packets();
  processor_state->handle_buffer_completion(true);
  EXPECT_EQ(processor_state->packets_missing, 12);

  typename TestMultipleFPGAWithOctetConfig::InputPacketSamplesType *samples =
      (typename TestMultipleFPGAWithOctetConfig::InputPacketSamplesType *)
          mock_pipeline->last_packet_data->get_samples_ptr();

  for (int channel = 0; channel < TestMultipleFPGAWithOctetConfig::NR_CHANNELS;
       channel++) {
    for (int receiver = 0;
         receiver < TestMultipleFPGAWithOctetConfig::NR_RECEIVERS_PER_PACKET;
         receiver++) {
      for (int pkt = -1;
           pkt <
           static_cast<int>(
               TestMultipleFPGAWithOctetConfig::NR_PACKETS_FOR_CORRELATION + 1);
           pkt++) {
        for (int fpga = 0;
             fpga < TestMultipleFPGAWithOctetConfig::NR_FPGA_SOURCES; fpga++) {
          for (int t = 0;
               t < TestMultipleFPGAWithOctetConfig::NR_TIME_STEPS_PER_PACKET;
               t++) {
            for (int pol = 0;
                 pol < TestMultipleFPGAWithOctetConfig::NR_POLARIZATIONS;
                 pol++) {
              std::complex<int8_t> expected_value;

              if ((fpga == 0 &&
                   (pkt == -1 ||
                    pkt == static_cast<int>(TestMultipleFPGAConfig::
                                                NR_PACKETS_FOR_CORRELATION))) ||
                  (fpga == 1 &&
                   pkt >=
                       static_cast<int>(
                           TestMultipleFPGAConfig::NR_PACKETS_FOR_CORRELATION) -
                           2) ||

                  (fpga == 2 && pkt >= 7) ||

                  (fpga == 3 && pkt >= 9)

              ) {
                expected_value = {static_cast<int8_t>(0),
                                  static_cast<int8_t>(0)};
              } else {
                expected_value = {static_cast<int8_t>(fpga + 1),
                                  static_cast<int8_t>(fpga + 1)};
              }

              EXPECT_EQ(samples[0][channel][pkt + 1][fpga][t][receiver][pol],
                        expected_value)

                  << "Mismatch at channel " << channel << " pkt " << pkt
                  << " fpga " << fpga << " t " << t << " receiver " << receiver
                  << " pol" << pol << std::endl;
            }
          }
        }
      }
    }
  }
}
