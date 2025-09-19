#include <arpa/inet.h>
#include <pcap/pcap.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define UDP_PORT 12345
#define MIN_PCAP_HEADER_SIZE 58

// Your custom headers
#pragma pack(push, 1)
struct EthernetHeader {
  uint8_t dst[6];
  uint8_t src[6];
  uint16_t ethertype;
};

struct IPHeader {
  uint8_t version_ihl;
  uint8_t dscp_ecn;
  uint16_t total_length;
  uint16_t identification;
  uint16_t flags_fragment;
  uint8_t ttl;
  uint8_t protocol;
  uint16_t header_checksum;
  uint32_t src_ip;
  uint32_t dst_ip;
};

struct UDPHeader {
  uint16_t src_port;
  uint16_t dst_port;
  uint16_t length;
  uint16_t checksum;
};

struct CustomHeader {
  uint64_t sample_count;
  uint32_t fpga_id;
  uint16_t freq_channel;
  uint8_t padding[8];
};
#pragma pack(pop)

struct PacketInfo {
  uint64_t sample_count;
  uint16_t freq_channel;
  uint32_t fpga_id;
  int payload_size;
};

PacketInfo get_packet_info(const u_char *packet, const int size) {
  PacketInfo info = {0};

  if (size < MIN_PCAP_HEADER_SIZE) {
    printf("Packet too small (%d bytes)\n", size);
    return info;
  }

  // Parse headers
  const EthernetHeader *eth = (const EthernetHeader *)packet;
  if (ntohs(eth->ethertype) != 0x0800) {
    printf("Not IPv4 packet\n");
    return info;
  }

  const IPHeader *ip = (const IPHeader *)(packet + 14);
  if ((ip->version_ihl >> 4) != 4) {
    printf("Not IPv4\n");
    return info;
  }

  const UDPHeader *udp = (const UDPHeader *)(packet + 34);
  const CustomHeader *custom = (const CustomHeader *)(packet + 42);

  info.sample_count = custom->sample_count;
  info.freq_channel = custom->freq_channel;
  info.fpga_id = custom->fpga_id;
  info.payload_size = size - MIN_PCAP_HEADER_SIZE;

  return info;
}

int send_custom_packet(int sockfd, struct sockaddr_in *server_addr,
                       const u_char *packet_data, int packet_len) {

  PacketInfo info = get_packet_info(packet_data, packet_len);
  if (info.payload_size <= 0) {
    printf("No payload or invalid packet\n");
    return -1;
  }

  printf("Packet info: sample_count=%lu, freq_channel=%u, fpga_id=%u, "
         "payload_size=%d\n",
         info.sample_count, info.freq_channel, info.fpga_id, info.payload_size);

  // Send the entire packet (headers + payload)
  ssize_t sent = sendto(sockfd, packet_data, packet_len, 0,
                        (struct sockaddr *)server_addr, sizeof(*server_addr));

  if (sent < 0) {
    perror("sendto");
    return -1;
  }

  printf("Sent complete packet (%zd bytes)\n", sent);
  return 0;
}

int main(int argc, char *argv[]) {
  pcap_t *handle;
  char errbuf[PCAP_ERRBUF_SIZE];
  struct pcap_pkthdr *header;
  const u_char *packet;
  int sockfd;
  struct sockaddr_in server_addr;
  const char *server_ip = "127.0.0.1";
  int packet_count = 0;
  int custom_packet_count = 0;
  int res;

  if (argc < 2) {
    printf("Usage: %s <pcap_file>\n", argv[0]);
    return 1;
  }

  // Open pcap file using libpcap
  handle = pcap_open_offline(argv[1], errbuf);
  if (!handle) {
    fprintf(stderr, "pcap_open_offline failed: %s\n", errbuf);
    return 1;
  }

  printf("Reading pcap file: %s\n", argv[1]);
  printf("Sending packets to %s:%d\n\n", server_ip, UDP_PORT);

  // Create UDP socket for sending
  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0) {
    perror("socket");
    pcap_close(handle);
    return 1;
  }

  // Setup server address
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(UDP_PORT);
  server_addr.sin_addr.s_addr = inet_addr(server_ip);

  // Read packets using pcap_next_ex
  while ((res = pcap_next_ex(handle, &header, &packet)) >= 0) {
    if (res == 0)
      continue; // Timeout (shouldn't happen with offline files)

    packet_count++;
    printf("\n--- Packet #%d (%d bytes) ---\n", packet_count, header->len);

    // Process and send the packet
    if (send_custom_packet(sockfd, &server_addr, packet, header->len) == 0) {
      custom_packet_count++;
    }

    // Small delay between packets
    usleep(50); // 100us
  }

  if (res == -1) {
    fprintf(stderr, "Error reading packets: %s\n", pcap_geterr(handle));
  }

  printf("\n=== Summary ===\n");
  printf("Total packets read: %d\n", packet_count);
  printf("Custom packets sent: %d\n", custom_packet_count);

  close(sockfd);
  pcap_close(handle);
  return 0;
}
