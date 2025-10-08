#include "spatial/packet_formats.hpp"
#include "spatial/logging.hpp"
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdexcept>
#include <string>

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

LambdaFinalPacketData::LambdaFinalPacketData() {
  // allocate samples
  CUDA_CHECK(cudaHostAlloc((void **)&samples, sizeof(LambdaPacketSamples),
                           cudaHostAllocDefault));
  // allocate scales
  CUDA_CHECK(cudaHostAlloc((void **)&scales, sizeof(LambdaScales),
                           cudaHostAllocDefault));
};

LambdaFinalPacketData::~LambdaFinalPacketData() {
  cudaFreeHost(samples);
  cudaFreeHost(scales);
};

void LambdaFinalPacketData::zero_missing_packets() {
  for (auto i = 0; i < NR_LAMBDA_CHANNELS; ++i) {
    for (auto j = 0; j < NR_LAMBDA_PACKETS_FOR_CORRELATION; ++j) {
      for (auto k = 0; k < NR_LAMBDA_FPGAS; ++k) {
        if (arrivals[i][j][k] == 0) {
          for (auto m = 0; m < NR_LAMBDA_RECEIVERS_PER_PACKET; ++m) {
            for (auto n = 0; n < NR_LAMBDA_POLARIZATIONS; ++n) {
              *scales[i][j][k * NR_LAMBDA_RECEIVERS_PER_PACKET + m][n] = 0;
            }
          }
        }
      }
    }
  }
};
