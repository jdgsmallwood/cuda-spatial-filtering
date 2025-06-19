#include "spatial/ethernet.hpp"
#include <stdexcept>

#ifndef MIN_PCAP_HEADER_SIZE
// minimum header size (14+20+8+16)
#define MIN_PCAP_HEADER_SIZE 58
#endif

PacketInfo get_packet_info(const u_char *packet, const int size) {
  if (size < MIN_PCAP_HEADER_SIZE) {
    printf("Packet too small\n");
    throw std::runtime_error("Packet size too small!");
  }

  const CustomHeader *custom =
      reinterpret_cast<const CustomHeader *>(packet + 42);

  PacketInfo p = {custom->sample_count, custom->freq_channel};
  return p;
}
