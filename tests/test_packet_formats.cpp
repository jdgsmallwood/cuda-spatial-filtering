#include "spatial/packet_formats.hpp"
#include <arpa/inet.h> // for htons, htonl
#include <cstring>     // for memcpy, memset
#include <gtest/gtest.h>

constexpr size_t HEADER_SIZE = sizeof(EthernetHeader) + sizeof(IPHeader) +
                               sizeof(UDPHeader) + sizeof(CustomHeader);

template <typename T>
typename T::PacketEntryType
create_valid_test_packet(const int sample_count, const int fpga_id,
                         const int channel, const int src_ip_octet3 = 0) {
  typename T::PacketEntryType entry;
  memset(&entry, 0, sizeof(entry)); // Zero everything

  uint8_t *ptr = entry.data;

  // 1. Ethernet Header
  EthernetHeader eth = {};
  std::memcpy(eth.dst, "\xaa\xbb\xcc\xdd\xee\xff", 6);
  std::memcpy(eth.src, "\x11\x22\x33\x44\x55\x66", 6);
  eth.ethertype = htons(0x0800); // IPv4

  std::memcpy(ptr, &eth, sizeof(eth));
  ptr += sizeof(eth);

  // 2. IP Header
  IPHeader ip = {};
  ip.version_ihl = (4 << 4) | 5; // IPv4, IHL=5
  ip.total_length =
      htons(sizeof(IPHeader) + sizeof(UDPHeader) + sizeof(CustomHeader) +
            sizeof(typename T::PacketPayloadType));
  ip.protocol = 17; // UDP
                    // 1. Start with the full host-order IP address (10.0.0.1)
  uint32_t host_ip =
      0x0a000001; // This is (10 << 24) | (0 << 16) | (0 << 8) | 1

  // 2. Clear the current 3rd octet (bits 8-15) and set the new one
  // The mask ~0x0000FF00U clears the 3rd octet.
  host_ip &= ~0x0000FF00U;

  // 3. Set the new 3rd octet value
  host_ip |= (static_cast<uint32_t>(src_ip_octet3) << 8);

  // 4. Convert the final value to network byte order
  ip.src_ip = htonl(host_ip);
  ip.dst_ip = htonl(0x0a000002); // 10.0.0.2
  std::memcpy(ptr, &ip, sizeof(ip));
  ptr += sizeof(ip);

  // 3. UDP Header
  UDPHeader udp = {};
  udp.src_port = htons(12345);
  udp.dst_port = htons(54321);
  udp.length = htons(sizeof(UDPHeader) + sizeof(CustomHeader) +
                     sizeof(typename T::PacketPayloadType));

  std::memcpy(ptr, &udp, sizeof(udp));
  ptr += sizeof(udp);

  // 4. Custom Header
  CustomHeader custom = {};
  custom.sample_count = sample_count;
  custom.fpga_id = fpga_id;
  custom.freq_channel = channel;

  std::memcpy(ptr, &custom, sizeof(custom));
  ptr += sizeof(custom);

  // 5. Payload
  typename T::PacketPayloadType payload = {};
  // Fill scales with incrementing numbers
  for (int i = 0; i < T::NR_RECEIVERS; ++i) {
    for (int j = 0; j < T::NR_POLARIZATIONS; ++j) {
      payload.scales[i][j] = i * 10;
    }
  }

  // Fill complex data with real = row, imag = col
  for (int t = 0; t < T::NR_TIME_STEPS_PER_PACKET; ++t) {
    for (int r = 0; r < T::NR_RECEIVERS; ++r) {
      for (int j = 0; j < T::NR_POLARIZATIONS; ++j) {
        payload.data[t][r][j] = std::complex<int8_t>(t, r);
      }
    }
  }

  std::memcpy(ptr, &payload, sizeof(payload));
  ptr += sizeof(payload);

  struct sockaddr_in sender_addr = {};
  sender_addr.sin_family = AF_INET;
  sender_addr.sin_addr.s_addr = htonl(host_ip);

  // Finalize
  entry.length = ptr - entry.data;
  entry.timestamp = {1234567890, 123456}; // fake timestamp
  entry.processed = false;
  entry.sender_addr = sender_addr;

  return entry;
}

TEST(PacketFormatTests, TestValidTestPacketSize) {
  using Config = LambdaConfig<8, 1, 64, 10, 2, 10, 1, 1, 32, 32, 10000>;
  Config::PacketEntryType test_packet =
      create_valid_test_packet<Config>(1, 1, 1);
  ASSERT_EQ(test_packet.length, 2664);
}

TEST(PacketFormatTests, TestLambdaPacketEntryParsedFormat) {
  using Config = LambdaConfig<8, 1, 64, 10, 2, 10, 1, 1, 32, 32, 10000>;
  int sample_count = 1;
  int fpga_id = 2;
  int channel = 3;

  Config::PacketEntryType test_packet =
      create_valid_test_packet<Config>(sample_count, fpga_id, channel);

  ProcessedPacket<Config::PacketScaleStructure, Config::PacketDataStructure>
      processed_packet = test_packet.parse();

  ASSERT_EQ(processed_packet.sample_count, sample_count);
  ASSERT_EQ(processed_packet.fpga_id, fpga_id);
  ASSERT_EQ(processed_packet.freq_channel, channel);

  ASSERT_EQ(processed_packet.payload->scales[0][0], 0);
  ASSERT_EQ(processed_packet.payload->scales[1][0], 10);

  ASSERT_EQ(processed_packet.payload->data[0][0][0],
            std::complex<int8_t>(0, 0));
}

TEST(PacketFormatTests, TestIPThirdOctetFPGAIDParsing) {
  // When ThirdOctetFPGAIDParsing is used the FPGA ID will be equal to the
  // third octet i.e. 10.0.5.10 = 5
  // regardless of what is put in the fpga_id header.
  using Config = LambdaConfig<8, 1, 64, 10, 2, 10, 1, 1, 32, 32, 10000, true>;
  int sample_count = 1;
  int fpga_id = 100;
  int channel = 4;
  int desired_fpga_id = 5;

  Config::PacketEntryType test_packet = create_valid_test_packet<Config>(
      sample_count, fpga_id, channel, desired_fpga_id);

  ProcessedPacket<Config::PacketScaleStructure, Config::PacketDataStructure>
      processed_packet = test_packet.parse();
  ASSERT_EQ(processed_packet.fpga_id, desired_fpga_id);
}
