#pragma once
#include "spatial/logging.hpp"
#include <complex>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <netinet/in.h>
#include <sys/time.h>
#include <unistd.h>
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

struct FinalPacketData {
  size_t start_seq_id;
  size_t end_seq_id;
  size_t buffer_index;

  virtual void *get_samples_ptr() = 0;
  virtual size_t get_samples_elements_size() = 0;

  virtual void *get_scales_ptr() = 0;
  virtual size_t get_scales_element_size() = 0;

  virtual bool *get_arrivals_ptr() = 0;
  virtual size_t get_arrivals_size() = 0;

  virtual void zero_missing_packets() = 0;
  virtual int get_num_missing_packets() = 0;
};

// This one needs to be like this because it will be defined in the
// PacketStructure struct.
template <typename PacketSamplesType, typename PacketScalesType,
          size_t NR_CHANNELS, size_t NR_PACKETS_FOR_CORRELATION,
          size_t NR_RECEIVERS, size_t NR_POLARIZATIONS, size_t NR_FPGAS>
struct LambdaFinalPacketData : public FinalPacketData {
  using ArrivalsType = bool[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION][NR_FPGAS];
  PacketSamplesType *samples = nullptr;
  PacketScalesType *scales = nullptr;
  ArrivalsType *arrivals = nullptr;

  void *get_samples_ptr() override { return (void *)samples; };
  void *get_scales_ptr() override { return (void *)scales; };
  bool *get_arrivals_ptr() override { return (bool *)arrivals; };

  size_t get_samples_elements_size() override {
    return sizeof(PacketSamplesType);
  };
  size_t get_scales_element_size() override {
    return sizeof(PacketScalesType);
  };

  size_t get_arrivals_size() override { return sizeof(ArrivalsType); };

  void zero_missing_packets() override {
    for (auto i = 0; i < NR_CHANNELS; ++i) {
      for (auto j = 0; j < NR_PACKETS_FOR_CORRELATION; ++j) {
        for (auto k = 0; k < NR_FPGAS; ++k) {
          if (arrivals[0][i][j][k] == 0) {
            for (auto m = 0; m < NR_RECEIVERS; ++m) {
              for (auto n = 0; n < NR_POLARIZATIONS; ++n) {
                scales[0][i][j][k * NR_RECEIVERS + m][n] = 0;
              }
            }
          }
        }
      }
    }
  };
  int get_num_missing_packets() override {
    int sum = 0;
    for (auto i = 0; i < NR_CHANNELS; ++i) {
      for (auto j = 0; j < NR_PACKETS_FOR_CORRELATION; ++j) {
        for (auto k = 0; k < NR_FPGAS; ++k) {
          if (arrivals[0][i][j][k] == 0) {
            sum++;
          }
        }
      }
    }
    return sum;
  };
  LambdaFinalPacketData() {

    // allocate samples
    CUDA_CHECK(cudaHostAlloc((void **)&samples, sizeof(PacketSamplesType),
                             cudaHostAllocDefault));
    // allocate scales
    CUDA_CHECK(cudaHostAlloc((void **)&scales, sizeof(PacketScalesType),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&arrivals, sizeof(ArrivalsType),
                             cudaHostAllocDefault));
  };
  ~LambdaFinalPacketData() {
    cudaFreeHost(samples);
    cudaFreeHost(scales);
    cudaFreeHost(arrivals);
  };
};

template <typename PacketScaleStructure, typename PacketDataStructure>
struct PacketPayload {
  PacketScaleStructure scales;
  PacketDataStructure data;
};

// Processed packet info
template <typename PacketScaleStructure, typename PacketDataStructure>
struct ProcessedPacket {
  uint64_t sample_count;
  uint32_t fpga_id;
  uint16_t freq_channel;
  PacketPayload<PacketScaleStructure, PacketDataStructure> *payload;
  int payload_size;
  struct timeval timestamp;
  bool *original_packet_processed;
};

// Packet storage for ring buffer
template <typename PacketScaleStructure, typename PacketDataStructure>
struct PacketEntry {
  uint8_t data[BUFFER_SIZE];
  int length;
  struct sockaddr_in sender_addr;
  struct timeval timestamp;
  bool processed; // 0 = unprocessed, 1 = processed

  virtual ProcessedPacket<PacketScaleStructure, PacketDataStructure>
  parse() = 0;
};

template <typename PacketScaleStructure, typename PacketDataStructure>
struct LambdaPacketEntry
    : public PacketEntry<PacketScaleStructure, PacketDataStructure> {
  ProcessedPacket<PacketScaleStructure, PacketDataStructure> parse() override {

    LOG_DEBUG("Entering parser...\n");
    ProcessedPacket<PacketScaleStructure, PacketDataStructure> result = {0};

    if (this->length < MIN_PCAP_HEADER_SIZE) {
      LOG_WARN("Packet too small for custom headers\n");
      this->processed = true;
      return result;
    }

    const EthernetHeader *eth = (const EthernetHeader *)this->data;
    if (ntohs(eth->ethertype) != 0x0800) {
      LOG_ERROR("Not IPv4 packet\n");
      return result;
    }

    const CustomHeader *custom = (const CustomHeader *)(this->data + 42);

    result.sample_count = custom->sample_count;
    result.fpga_id = custom->fpga_id;
    result.freq_channel = custom->freq_channel;
    result.timestamp = this->timestamp;

    // Point to payload (after headers)
    result.payload = reinterpret_cast<
        PacketPayload<PacketScaleStructure, PacketDataStructure> *>(
        this->data + MIN_PCAP_HEADER_SIZE);
    result.payload_size = this->length - MIN_PCAP_HEADER_SIZE;
    // This adds a pointer to the original processed boolean on the queue
    // so that we can update it later once it's been processed.
    result.original_packet_processed = &this->processed;

    return result;
  };
};

template <size_t NR_CHANNELS_T, size_t NR_FPGA_SOURCES_T,
          size_t NR_TIME_STEPS_PER_PACKET_T, size_t NR_RECEIVERS_T,
          size_t NR_POLARIZATIONS_T, size_t NR_RECEIVERS_PER_PACKET_T,
          size_t NR_PACKETS_FOR_CORRELATION_T, size_t NR_BEAMS_T,
          size_t NR_PADDED_RECEIVERS_T, size_t NR_PADDED_RECEIVERS_PER_BLOCK_T,
          size_t NR_CORRELATED_BLOCKS_TO_ACCUMULATE_T>
struct LambdaConfig {

  static constexpr size_t NR_CHANNELS = NR_CHANNELS_T;
  static constexpr size_t NR_FPGA_SOURCES = NR_FPGA_SOURCES_T;
  static constexpr size_t NR_TIME_STEPS_PER_PACKET = NR_TIME_STEPS_PER_PACKET_T;
  static constexpr size_t NR_RECEIVERS = NR_RECEIVERS_T;
  static constexpr size_t NR_POLARIZATIONS = NR_POLARIZATIONS_T;
  static constexpr size_t NR_RECEIVERS_PER_PACKET = NR_RECEIVERS_PER_PACKET_T;
  static constexpr size_t NR_PACKETS_FOR_CORRELATION =
      NR_PACKETS_FOR_CORRELATION_T;
  static constexpr size_t NR_BEAMS = NR_BEAMS_T;
  static constexpr size_t NR_PADDED_RECEIVERS = NR_PADDED_RECEIVERS_T;
  static constexpr size_t NR_PADDED_RECEIVERS_PER_BLOCK =
      NR_PADDED_RECEIVERS_PER_BLOCK_T;
  static constexpr size_t NR_CORRELATED_BLOCKS_TO_ACCUMULATE =
      NR_CORRELATED_BLOCKS_TO_ACCUMULATE_T;
  static constexpr size_t NR_BASELINES =
      NR_PADDED_RECEIVERS * (NR_PADDED_RECEIVERS + 1) / 2;
  static constexpr size_t NR_BASELINES_UNPADDED =
      NR_RECEIVERS * (NR_RECEIVERS + 1) / 2;
  static constexpr size_t COMPLEX = 2;

  template <typename T, int RECEIVERS = NR_RECEIVERS>
  using LambdaPacketSamplesT =
      std::complex<T>[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION]
                     [NR_TIME_STEPS_PER_PACKET][RECEIVERS][NR_POLARIZATIONS];
  template <typename T, int RECEIVERS = NR_RECEIVERS>
  using LambdaPacketSamplesPlanarT =
      T[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION][NR_TIME_STEPS_PER_PACKET]
       [RECEIVERS][NR_POLARIZATIONS][2]; // for complex

  using PacketScalesType = int16_t[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION]
                                  [NR_RECEIVERS][NR_POLARIZATIONS];
  using Sample = std::complex<int8_t>;
  using PacketSamplesType = LambdaPacketSamplesT<int8_t>;
  using HalfPacketSamplesType = LambdaPacketSamplesT<__half>;
  using PaddedPacketSamplesType =
      LambdaPacketSamplesT<__half, NR_PADDED_RECEIVERS>;
  using PacketSamplesPlanarType = LambdaPacketSamplesPlanarT<int8_t>;
  using HalfPacketSamplesPlanarType = LambdaPacketSamplesPlanarT<__half>;
  using PaddedPacketSamplesPlanarType =
      LambdaPacketSamplesPlanarT<__half, NR_PADDED_RECEIVERS>;
  using PacketScaleStructure =
      int16_t[NR_RECEIVERS_PER_PACKET][NR_POLARIZATIONS];
  using PacketDataStructure =
      std::complex<int8_t>[NR_TIME_STEPS_PER_PACKET][NR_RECEIVERS_PER_PACKET]
                          [NR_POLARIZATIONS];
  using PacketPayloadType =
      PacketPayload<PacketScaleStructure, PacketDataStructure>;
  using ProcessedPacketType =
      ProcessedPacket<PacketScaleStructure, PacketDataStructure>;
  using PacketEntryType =
      LambdaPacketEntry<PacketScaleStructure, PacketDataStructure>;
  using PacketFinalDataType =
      LambdaFinalPacketData<PacketSamplesType, PacketScalesType, NR_CHANNELS,
                            NR_PACKETS_FOR_CORRELATION, NR_RECEIVERS,
                            NR_POLARIZATIONS, NR_FPGA_SOURCES>;
  using BeamOutputType =
      float[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS]
           [NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET][COMPLEX];
  using ArrivalsOutputType =
      bool[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION][NR_FPGA_SOURCES];
  using VisibilitiesOutputType =
      float[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS]
           [COMPLEX];
};
