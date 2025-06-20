#include "spatial/ethernet.hpp"
#include <pcap/pcap.h>
#include <stdexcept>
#include <vector>
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
void process_packet(const u_char *packet, const int size, Samples *agg_samples,
                    std::vector<Tscale> &scales_output, const int start_seq_id,
                    const int start_freq, const int nr_time_steps_per_packet,
                    const int nr_blocks_for_correlation,
                    const int nr_times_per_block,
                    const int nr_actual_receivers) {
  if (size < 58) { // minimum header size (14+20+8+16)
    printf("Packet too small\n");
    return;
  }

  const EthernetHeader *eth = reinterpret_cast<const EthernetHeader *>(packet);
  uint16_t ethertype = ntohs(eth->ethertype);
  if (ethertype != 0x0800) {
    printf("Not IPv4, skipping\n");
    return;
  }

  const IPHeader *ip = reinterpret_cast<const IPHeader *>(packet + 14);
  if ((ip->version_ihl >> 4) != 4) {
    printf("Not IPv4, skipping\n");
    return;
  }

  const UDPHeader *udp = reinterpret_cast<const UDPHeader *>(packet + 34);
  const CustomHeader *custom =
      reinterpret_cast<const CustomHeader *>(packet + 42);

  const int packet_num =
      (custom->sample_count - start_seq_id) / nr_time_steps_per_packet;
  const int packet_num_in_frame = packet_num % nr_blocks_for_correlation;
  const int sample_frame_to_populate = packet_num / nr_blocks_for_correlation;
  const int num_blocks_per_packet =
      nr_time_steps_per_packet / nr_times_per_block; // 8
  printf("Packet number is %u. Seq is %u and start_seq was %u\n", packet_num,
         custom->sample_count, start_seq_id);
  printf("Will write to block %u in frame %u\n", packet_num_in_frame,
         sample_frame_to_populate);
  printf("Num blocks per packet is %u\n", num_blocks_per_packet);
  // Scale factors start at offset 64
  const Tscale *scales = reinterpret_cast<const Tscale *>(packet + 64);
  for (auto j = 0; j < nr_actual_receivers; ++j) {
    scales_output[j] = scales[j];
  }

  // Filterbank samples after scales
  const Tin *samples = reinterpret_cast<const Tin *>(
      packet + 64 + nr_actual_receivers * sizeof(Tscale));
  const int freq_idx = custom->freq_channel - start_freq;
  for (auto k = 0; k < num_blocks_per_packet; ++k) {
    int pkt_idx = num_blocks_per_packet * packet_num_in_frame + k;
    for (auto i = 0; i < nr_times_per_block; ++i) {
      for (auto j = 0; j < nr_actual_receivers; ++j) {
        agg_samples[sample_frame_to_populate][freq_idx][pkt_idx][j][0][i] =
            Sample(samples[2 * k * nr_times_per_block * nr_actual_receivers +
                           2 * nr_actual_receivers * i + 2 * j],
                   samples[2 * k * nr_times_per_block * nr_actual_receivers +
                           2 * nr_actual_receivers * i + 2 * j + 1]);
      }
    }
  }
}
