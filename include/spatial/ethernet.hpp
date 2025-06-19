#pragma once
#include <cstdint>

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