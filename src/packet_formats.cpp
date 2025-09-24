#include "spatial/packet_formats.hpp"

ProcessedPacket LambdaPacketEntry::parse() {

  ProcessedPacket result = {0};

  if (length < MIN_PCAP_HEADER_SIZE) {
    printf("Packet too small for custom headers\n");
    processed = true;
    return result;
  }

  // Parse your custom packet structure
  const EthernetHeader *eth = (const EthernetHeader *)pkt->data;
  if (ntohs(eth->ethertype) != 0x0800) {
    printf("Not IPv4 packet\n");
    return result;
  }

  const CustomHeader *custom = (const CustomHeader *)(pkt->data + 42);

  result.sample_count = custom->sample_count;
  result.fpga_id = custom->fpga_id;
  result.freq_channel = custom->freq_channel;
  result.timestamp = pkt->timestamp;

  // Point to payload (after headers)
  result.payload =
      reinterpret_cast<PacketPayload *>(pkt->data + MIN_PCAP_HEADER_SIZE);
  result.payload_size = pkt->length - MIN_PCAP_HEADER_SIZE;
  result.original_packet_processed = &pkt->processed;

  return result;
}
