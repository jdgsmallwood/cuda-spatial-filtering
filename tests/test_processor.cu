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
constexpr static size_t NR_BUFFERS = 3;
// Simple mock pipeline that just tracks what it receives
class SimpleMockPipeline : public GPUPipeline {
public:
  std::atomic<int> execute_count{0};
  std::atomic<int> release_count{0};
  std::vector<int> buffer_indices_received;
  std::vector<unsigned long long> start_seqs_received;
  std::vector<unsigned long long> end_seqs_received;
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

  void dump_visibilities(const unsigned long long end_seq_num = 0) override {}

  int get_execute_count() const { return execute_count.load(); }
  int get_release_count() const { return release_count.load(); }
};

// Test fixture
class ProcessorStateTest : public ::testing::Test {
protected:
  ProcessorStateBase *processor_state;
  SimpleMockPipeline *mock_pipeline;

  void SetUp() override {
    processor_state = new ProcessorState<TestConfig, NR_BUFFERS>(
        10,                                   // nr_packets_for_correlation
        TestConfig::NR_TIME_STEPS_PER_PACKET, // nr_between_samples
        0                                     // min_freq_channel
    );

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
                                    uint16_t freq_channel, int val) {
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
          EXPECT_EQ(packet_samples[0][channel][packet_number][0][t][r]
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
                          uint16_t freq_channel, int val = 1) {
    create_lambda_packet(sample_count, fpga_id, freq_channel, val);
    processor_state->get_next_write_pointer();
  }
};

class ProcessorStateMultipleFPGATest : public ProcessorStateTest {

  void SetUp() override {
    processor_state = new ProcessorState<TestMultipleFPGAConfig, NR_BUFFERS>(
        10, // nr_packets_for_correlation
        TestMultipleFPGAConfig::NR_TIME_STEPS_PER_PACKET, // nr_between_samples
        0                                                 // min_freq_channel
    );

    mock_pipeline = new SimpleMockPipeline();
    mock_pipeline->set_state(processor_state);
    processor_state->set_pipeline(mock_pipeline);
    processor_state->synchronous_pipeline = true;
  }

  void create_lambda_packet(uint64_t sample_count, uint32_t fpga_id,
                            uint16_t freq_channel, int val) {
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
        payload->scales[r][p] = static_cast<int16_t>(1 + r); // Non-zero scales
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
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    processor_state->add_received_packet_metadata(total_length, addr);
  }
};

TEST_F(ProcessorStateTest, ProcessSinglePacketTest) {
  add_packet(1000, 0, 0);

  processor_state->process_all_available_packets();

  EXPECT_EQ(processor_state->packets_processed, 1);
}

TEST_F(ProcessorStateTest, BufferInitializationTest) {
  EXPECT_FALSE(processor_state->buffers_initialized);

  add_packet(1000, 0, 0);

  processor_state->process_all_available_packets();

  EXPECT_TRUE(processor_state->buffers_initialized);
}

TEST_F(ProcessorStateTest, FillOneBufferTest) {
  const uint64_t start_sample = 1000;

  // For one complete buffer, we need packets from all channels and all FPGAs
  // Total packets = NR_CHANNELS * NR_FPGA_SOURCES * NR_PACKETS_FOR_CORRELATION
  int total_packets = TestConfig::NR_CHANNELS * TestConfig::NR_FPGA_SOURCES *
                      TestConfig::NR_PACKETS_FOR_CORRELATION;

  for (int channel = 0; channel < TestConfig::NR_CHANNELS; channel++) {
    for (int fpga = 0; fpga < TestConfig::NR_FPGA_SOURCES; fpga++) {
      for (int pkt = 0; pkt < TestConfig::NR_PACKETS_FOR_CORRELATION; pkt++) {
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
  EXPECT_TRUE(processor_state->buffers_initialized);

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
    processor_state->process_all_available_packets();
    processor_state->handle_buffer_completion();
  }
  // add two packets that are way further along, this will cause
  // the pipeline to run w/ missing packets.
  add_packet(20000, 0, 0);
  add_packet(20000, 0, 1);
  processor_state->process_all_available_packets();
  processor_state->handle_buffer_completion();
  EXPECT_EQ(processor_state->packets_missing, 20);

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
  processor_state->handle_buffer_completion();
  EXPECT_EQ(processor_state->packets_missing, 0);

  typename TestMultipleFPGAConfig::InputPacketSamplesType *samples =
      (typename TestMultipleFPGAConfig::InputPacketSamplesType *)
          mock_pipeline->last_packet_data->get_samples_ptr();

  for (int channel = 0; channel < TestMultipleFPGAConfig::NR_CHANNELS;
       channel++) {
    for (int receiver = 0;
         receiver < TestMultipleFPGAConfig::NR_RECEIVERS_PER_PACKET;
         receiver++) {
      for (int pkt = 0;
           pkt < TestMultipleFPGAConfig::NR_PACKETS_FOR_CORRELATION; pkt++) {
        for (int fpga = 0; fpga < TestMultipleFPGAConfig::NR_FPGA_SOURCES;
             fpga++) {
          for (int t = 0; t < TestMultipleFPGAConfig::NR_TIME_STEPS_PER_PACKET;
               t++) {
            for (int pol = 0; pol < TestMultipleFPGAConfig::NR_POLARIZATIONS;
                 pol++) {
              std::complex<int8_t> expected_value;
              expected_value = {static_cast<int8_t>(fpga + 1),
                                static_cast<int8_t>(fpga + 1)};
              EXPECT_EQ(samples[0][channel][pkt][fpga][t][receiver][pol],
                        expected_value);
            }
          }
        }
      }
    }
  }
}
