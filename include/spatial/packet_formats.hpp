#pragma once
#include "spatial/logging.hpp"
#include <complex>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
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
  uint64_t start_seq_id;
  uint64_t end_seq_id;
  size_t buffer_index;

  virtual void *get_samples_ptr() = 0;
  virtual size_t get_samples_elements_size() = 0;

  virtual int *get_arrivals_ptr() = 0;
  virtual size_t get_arrivals_size() = 0;

  virtual void zero_samples() = 0;
  virtual void zero_arrivals() = 0;
  virtual size_t get_num_missing_packets() = 0;
};

// This one needs to be like this because it will be defined in the
// PacketStructure struct.
template <typename PacketSamplesType, size_t NR_CHANNELS,
          size_t NR_PACKETS_FOR_CORRELATION, size_t NR_RECEIVERS_PER_PACKET,
          size_t NR_POLARIZATIONS, size_t NR_FPGAS,
          size_t NR_TIME_STEPS_PER_PACKET>
struct LambdaFinalPacketData : public FinalPacketData {
  using ArrivalsType = int[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION][NR_FPGAS];
  PacketSamplesType *samples = nullptr;
  ArrivalsType *arrivals = nullptr;

  void *get_samples_ptr() override { return (void *)samples; };
  int *get_arrivals_ptr() override { return (int *)arrivals; };

  size_t get_samples_elements_size() override {
    return sizeof(PacketSamplesType);
  };

  size_t get_arrivals_size() override { return sizeof(ArrivalsType); };

  void zero_samples() override {
    std::memset(get_samples_ptr(), 0, get_samples_elements_size());
  };

  void zero_arrivals() override {

    std::memset(arrivals, 0, sizeof(ArrivalsType));
  }
  size_t get_num_missing_packets() override {
    size_t sum = 0;
    for (auto i = 0; i < NR_CHANNELS; ++i) {
      for (auto j = 0; j < NR_PACKETS_FOR_CORRELATION; ++j) {
        for (auto k = 0; k < NR_FPGAS; ++k) {
          sum += NR_TIME_STEPS_PER_PACKET - arrivals[0][i][j][k];
        }
      }
    }
    size_t missing_packets =
        sum / NR_TIME_STEPS_PER_PACKET + (sum % NR_TIME_STEPS_PER_PACKET != 0);
    return missing_packets;
  };

  LambdaFinalPacketData() {

    // allocate samples
    CUDA_CHECK(cudaHostAlloc((void **)&samples, sizeof(PacketSamplesType),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&arrivals, sizeof(ArrivalsType),
                             cudaHostAllocDefault));
  };
  ~LambdaFinalPacketData() {
    cudaFreeHost(samples);
    cudaFreeHost(arrivals);
  };
};

template <typename PacketScaleStructure, typename PacketDataStructure>
struct PacketPayload {
  PacketScaleStructure scales;
  PacketDataStructure data;
};

// Processed packet info
template <typename PacketDataStructure> struct ProcessedPacket {
  uint64_t sample_count;
  uint64_t timestamp;
  const PacketDataStructure *payload;
  bool *original_packet_processed;
  uint32_t fpga_id;
  uint32_t payload_size;
  uint16_t freq_channel;
} __attribute__((aligned(64)));

// Packet storage for ring buffer
template <typename PacketDataStructure> struct PacketEntry {
  uint8_t data[BUFFER_SIZE];
  int length;
  struct sockaddr_in sender_addr;
  struct timeval timestamp;
  bool processed; // 0 = unprocessed, 1 = processed

  virtual ProcessedPacket<PacketDataStructure> parse() = 0;
};

template <typename PacketScaleStructure, typename InputPacketDataStructure,
          typename OutputPacketDataStructure, size_t NR_RECEIVERS_PER_PACKET,
          size_t NR_POLARIZATIONS, size_t NR_TIME_STEPS_PER_PACKET,
          bool OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET>
struct LambdaPacketEntry : public PacketEntry<OutputPacketDataStructure> {
  __attribute__((hot))
  __attribute__((flatten)) ProcessedPacket<OutputPacketDataStructure>
  parse() noexcept override {

    // LOG_DEBUG("Entering parser...\n");
    const int length = this->length;
    const uint8_t *__restrict__ base = this->data;
    uint32_t offset = 0;
    if (length > sizeof(PacketScaleStructure) +
                     sizeof(InputPacketDataStructure) + sizeof(CustomHeader)) {
      offset = 42;
    }
    __builtin_prefetch(base + offset, 0, 3);
    if (length < MIN_PCAP_HEADER_SIZE) [[unlikely]] {
      this->processed = (length < MIN_PCAP_HEADER_SIZE);
      return {};
    }

    const CustomHeader *__restrict__ custom =
        (const CustomHeader *)(base + offset);

    uint32_t fpga_id;
    if (OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET) {
      const uint8_t *ip_bytes =
          (const uint8_t *)&this->sender_addr.sin_addr.s_addr;
      uint8_t third_octet = ip_bytes[2];
      fpga_id = (uint32_t)third_octet;
    } else {
      fpga_id = custom->fpga_id;
    }

    unpack_packet_data(
        reinterpret_cast<const PacketPayload<PacketScaleStructure,
                                             InputPacketDataStructure> *>(
            base + offset + sizeof(CustomHeader)));

    return ProcessedPacket<OutputPacketDataStructure>{
        .sample_count = custom->sample_count,
        .timestamp =
            this->timestamp.tv_sec * 1000000ULL + this->timestamp.tv_usec,
        .payload = &this->output_data,
        .original_packet_processed = &this->processed,
        // for now - take the IP address as the fpga_id.
        // i.e. 10.0.3.10 = FPGA ID 3.
        .fpga_id = fpga_id, // custom->fpga_id,
        .payload_size =
            static_cast<uint32_t>(length - (offset + sizeof(CustomHeader))),
        .freq_channel = custom->freq_channel};
  };

  OutputPacketDataStructure output_data;
  inline std::complex<int> convert_and_scale(std::complex<int8_t> z,
                                             int scale) {
    return {static_cast<int>(z.real()) * scale,
            static_cast<int>(z.imag()) * scale};
  };

  std::complex<__half> convert_float_to_half(std::complex<int> z) {
    return {__float2half(static_cast<float>(z.real())),
            __float2half(static_cast<float>(z.imag()))};
  };

  void unpack_packet_data(
      const PacketPayload<PacketScaleStructure,
                          InputPacketDataStructure> __restrict__ *payload) {
    using clock = std::chrono::steady_clock;
    auto start = clock::now();

    std::array<int, NR_POLARIZATIONS * NR_RECEIVERS_PER_PACKET> scales;
    for (auto i = 0; i < NR_RECEIVERS_PER_PACKET; ++i) {
      for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
        scales[i * NR_POLARIZATIONS + j] =
            static_cast<int>(payload->scales[i][j]);
      }
    }

    for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
      for (auto i = 0; i < NR_RECEIVERS_PER_PACKET; ++i) {
        for (auto j = 0; j < NR_POLARIZATIONS; ++j) {

          std::complex<int8_t> data = payload->data[k][i][j];
          std::complex<int> data_int =
              convert_and_scale(data, scales[i * NR_POLARIZATIONS + j]);
          this->output_data[k][i][j] = convert_float_to_half(data_int);
        }
      }
    }

    auto end = clock::now();
    auto duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    LOG_INFO("Duration was {}ns", duration_ns);
  };
};

template <size_t NR_CHANNELS_T, size_t NR_FPGA_SOURCES_T,
          size_t NR_TIME_STEPS_PER_PACKET_T, size_t NR_RECEIVERS_T,
          size_t NR_POLARIZATIONS_T, size_t NR_RECEIVERS_PER_PACKET_T,
          size_t NR_PACKETS_FOR_CORRELATION_T, size_t NR_BEAMS_T,
          size_t NR_PADDED_RECEIVERS_T, size_t NR_PADDED_RECEIVERS_PER_BLOCK_T,
          size_t NR_CORRELATED_BLOCKS_TO_ACCUMULATE_T,
          bool OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET = false,
          size_t FFT_DOWNSAMPLE_FACTOR_T = 64>
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

  template <typename T>
  using LambdaInputPacketSamplesPlanarT =
      T[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION][NR_FPGA_SOURCES]
       [NR_TIME_STEPS_PER_PACKET][NR_RECEIVERS_PER_PACKET][NR_POLARIZATIONS][2];
  template <typename T, int RECEIVERS = NR_RECEIVERS>
  using LambdaPacketSamplesPlanarT =
      T[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION][NR_TIME_STEPS_PER_PACKET]
       [RECEIVERS][NR_POLARIZATIONS][2]; // for complex

  using PacketScalesType = int16_t[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION]
                                  [NR_RECEIVERS][NR_POLARIZATIONS];

  using Sample = std::complex<int8_t>;
  using InputPacketSamplesType =
      std::complex<__half>[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION]
                          [NR_FPGA_SOURCES][NR_TIME_STEPS_PER_PACKET]
                          [NR_RECEIVERS_PER_PACKET][NR_POLARIZATIONS];
  using InputPacketSamplesPlanarType = LambdaInputPacketSamplesPlanarT<__half>;
  using PacketSamplesType = LambdaPacketSamplesT<int8_t>;
  using HalfPacketSamplesType = LambdaPacketSamplesT<__half>;
  using HalfInputPacketSamplesPlanarType =
      LambdaInputPacketSamplesPlanarT<__half>;

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
  using OutputPacketDataStructure =
      std::complex<__half>[NR_TIME_STEPS_PER_PACKET][NR_RECEIVERS_PER_PACKET]
                          [NR_POLARIZATIONS];
  using PacketPayloadType =
      PacketPayload<PacketScaleStructure, PacketDataStructure>;
  using ProcessedPacketType = ProcessedPacket<OutputPacketDataStructure>;
  using PacketEntryType =
      LambdaPacketEntry<PacketScaleStructure, PacketDataStructure,
                        OutputPacketDataStructure, NR_RECEIVERS_PER_PACKET,
                        NR_POLARIZATIONS, NR_TIME_STEPS_PER_PACKET,
                        OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET>;
  using PacketFinalDataType =
      LambdaFinalPacketData<InputPacketSamplesType, NR_CHANNELS,
                            NR_PACKETS_FOR_CORRELATION, NR_RECEIVERS_PER_PACKET,
                            NR_POLARIZATIONS, NR_FPGA_SOURCES,
                            NR_TIME_STEPS_PER_PACKET>;
  using BeamOutputType =
      __half[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS]
            [NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET][COMPLEX];
  using ArrivalsOutputType =
      bool[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION][NR_FPGA_SOURCES];
  using VisibilitiesOutputType =
      float[NR_CHANNELS][NR_BASELINES_UNPADDED][NR_POLARIZATIONS]
           [NR_POLARIZATIONS][COMPLEX];
  using EigenvalueOutputType =
      float[NR_CHANNELS][NR_POLARIZATIONS][NR_POLARIZATIONS][NR_RECEIVERS];
  using EigenvectorOutputType =
      std::complex<float>[NR_CHANNELS][NR_POLARIZATIONS][NR_POLARIZATIONS]
                         [NR_RECEIVERS][NR_RECEIVERS];
  using FFTCUFFTPreprocessingType =
      __half2[NR_CHANNELS][NR_POLARIZATIONS][NR_RECEIVERS]
             [NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION];
  using FFTCUFFTInputType = float2[NR_RECEIVERS][NR_TIME_STEPS_PER_PACKET *
                                                 NR_PACKETS_FOR_CORRELATION];
  using FFTCUFFTOutputType = float2[NR_RECEIVERS][NR_TIME_STEPS_PER_PACKET *
                                                  NR_PACKETS_FOR_CORRELATION];

  using MultiChannelFFTCUFFTInputType =
      float2[NR_CHANNELS][NR_POLARIZATIONS][NR_RECEIVERS]
            [NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION];
  using MultiChannelFFTCUFFTOutputType =
      float2[NR_CHANNELS][NR_POLARIZATIONS][NR_RECEIVERS]
            [NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION];
  constexpr static int FFT_DOWNSAMPLE_FACTOR = FFT_DOWNSAMPLE_FACTOR_T;
  using FFTOutputType =
      float[NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION /
            FFT_DOWNSAMPLE_FACTOR];
  using AntennaFFTOutputType =
      float[NR_RECEIVERS][NR_TIME_STEPS_PER_PACKET *
                          NR_PACKETS_FOR_CORRELATION / FFT_DOWNSAMPLE_FACTOR];
  using MultiChannelAntennaFFTOutputType =
      float[NR_CHANNELS][NR_POLARIZATIONS][NR_RECEIVERS]
           [NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION /
            FFT_DOWNSAMPLE_FACTOR];
  using PulsarFoldOutputType = float[NR_CHANNELS][16][NR_POLARIZATIONS][256];
};
