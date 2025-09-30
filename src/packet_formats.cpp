#include "spatial/packet_formats.hpp"
#include "spatial/logging.hpp"

ProcessedPacket LambdaPacketEntry::parse() {
  LOG_INFO("Entering parser...\n");
  ProcessedPacket result = {0};

  if (length < MIN_PCAP_HEADER_SIZE) {
    LOG_WARN("Packet too small for custom headers\n");
    processed = true;
    return result;
  }

  // Parse your custom packet structure
  const EthernetHeader *eth = (const EthernetHeader *)data;
  if (ntohs(eth->ethertype) != 0x0800) {
    LOG_ERROR("Not IPv4 packet\n");
    return result;
  }

  const CustomHeader *custom = (const CustomHeader *)(data + 42);

  result.sample_count = custom->sample_count;
  result.fpga_id = custom->fpga_id;
  result.freq_channel = custom->freq_channel;
  result.timestamp = timestamp;

  // Point to payload (after headers)
  result.payload =
      reinterpret_cast<PacketPayload *>(data + MIN_PCAP_HEADER_SIZE);
  result.payload_size = length - MIN_PCAP_HEADER_SIZE;
  result.original_packet_processed = &processed;

  return result;
}
