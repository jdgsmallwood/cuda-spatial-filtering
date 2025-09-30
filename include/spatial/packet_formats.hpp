#pragma once
#include <cstdint>
#include <unistd.h>

#include <complex>
#include <netinet/in.h>
#include <sys/time.h>
#define BUFFER_SIZE 4096
#define MIN_PCAP_HEADER_SIZE 64

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

constexpr int NR_LAMBDA_CHANNELS = 8;
constexpr int NR_LAMBDA_PACKETS_FOR_CORRELATION = 16;
constexpr int NR_LAMBDA_TIME_STEPS_PER_PACKET = 64;
constexpr int NR_LAMBDA_ACTUAL_RECEIVERS = 20;
constexpr int NR_LAMBDA_RECEIVERS_PER_PACKET = 20;
constexpr int NR_LAMBDA_FPGAS = 1;

typedef std::complex<int8_t> LambdaSample;
typedef LambdaSample LambdaPacket[NR_LAMBDA_TIME_STEPS_PER_PACKET]
                                 [NR_LAMBDA_RECEIVERS_PER_PACKET];

typedef LambdaSample LambdaPacketSamples[NR_LAMBDA_CHANNELS]
                                        [NR_LAMBDA_PACKETS_FOR_CORRELATION]
                                        [NR_LAMBDA_TIME_STEPS_PER_PACKET]
                                        [NR_LAMBDA_ACTUAL_RECEIVERS];

struct FinalPacketData {
  size_t start_seq_id;
  size_t end_seq_id;
  size_t buffer_index;

  virtual void *get_samples_ptr() = 0;
  virtual size_t get_samples_elements_size() = 0;

  virtual void *get_scales_ptr() = 0;
  virtual size_t get_scales_element_size() = 0;

  virtual bool *get_arrivals_ptr() = 0;

  virtual void zero_missing_packets() = 0;
};

struct LambdaFinalPacketData : public FinalPacketData {
  LambdaPacketSamples samples;
  int16_t scales[NR_LAMBDA_CHANNELS][NR_LAMBDA_PACKETS_FOR_CORRELATION]
                [NR_LAMBDA_ACTUAL_RECEIVERS];
  bool arrivals[NR_LAMBDA_CHANNELS][NR_LAMBDA_PACKETS_FOR_CORRELATION]
               [NR_LAMBDA_FPGAS];

  void *get_samples_ptr() override { return (void *)&samples; };
  void *get_scales_ptr() override { return (void *)&scales; };
  bool *get_arrivals_ptr() override { return &arrivals[0][0][0]; };

  size_t get_samples_elements_size() { return sizeof(LambdaPacketSamples); };
  size_t get_scales_element_size() { return sizeof(scales); };

  void zero_missing_packets() override;
};

using PacketDataStructure =
    std::complex<int8_t>[NR_LAMBDA_TIME_STEPS_PER_PACKET]
                        [NR_LAMBDA_ACTUAL_RECEIVERS];

using PacketScaleStructure = int16_t[NR_LAMBDA_ACTUAL_RECEIVERS];

struct PacketPayload {
  PacketScaleStructure scales;
  PacketDataStructure data;
};

// Processed packet info
struct ProcessedPacket {
  uint64_t sample_count;
  uint32_t fpga_id;
  uint16_t freq_channel;
  PacketPayload *payload;
  int payload_size;
  struct timeval timestamp;
  bool *original_packet_processed;
};

// Packet storage for ring buffer
struct PacketEntry {
  uint8_t data[BUFFER_SIZE];
  int length;
  struct sockaddr_in sender_addr;
  struct timeval timestamp;
  bool processed; // 0 = unprocessed, 1 = processed

  virtual ProcessedPacket parse() = 0;
};

struct LambdaPacketEntry : public PacketEntry {
  ProcessedPacket parse() override;
};

struct LambdaPacketStructure {
  using Sample = LambdaSample;
  using PacketPayloadType = PacketPayload;
  using PacketSamplesType = LambdaPacketSamples;
  using ProcessedPacketType = ProcessedPacket;
  using Packet = LambdaPacket;
  using PacketEntryType = LambdaPacketEntry;
  using PacketFinalDataType = LambdaFinalPacketData;
};
