#pragma once

#include "spatial/packet_formats.hpp"
#include "spatial/spatial.hpp"

#include <complex>
#include <cstdint>
#include <cstring>
#include <netinet/in.h>

// Builders for synthetic on-the-wire LAMBDA packets (Ethernet + IP + UDP +
// CustomHeader + payload), consolidating the near-duplicate byte-layout logic
// that previously lived separately in tests/test_processor.cu
// (`create_lambda_packet`) and tests/test_packet_formats.cpp
// (`create_valid_test_packet`).
namespace test_support {

// Writes one wire-format packet for config `T` at `dest`, with sample/scale
// values supplied by caller-provided generators:
//   sample_fn(time_step, receiver, polarization) -> std::complex<int8_t>
//   scale_fn(receiver, polarization)             -> int16_t
//
// `src_ip_octet3` is encoded into the source IP's third octet -- this is what
// `LambdaConfig`'s `OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET` parsing path reads
// the FPGA id from (see TestIPThirdOctetFPGAIDParsing / LambdaPacketEntry).
//
// Returns the total packet length in bytes (Ethernet+IP+UDP+Custom+Payload),
// i.e. how much of `dest` was written.
template <typename T, typename SampleFn, typename ScaleFn>
size_t build_lambda_wire_packet(uint8_t *dest, uint64_t sample_count,
                                uint32_t fpga_id, uint16_t freq_channel,
                                SampleFn &&sample_fn, ScaleFn &&scale_fn,
                                uint8_t src_ip_octet3 = 0) {
  uint8_t *data_ptr = dest;

  EthernetHeader *eth = reinterpret_cast<EthernetHeader *>(data_ptr);
  std::memset(eth, 0, sizeof(EthernetHeader));
  eth->ethertype = htons(0x0800); // IPv4
  data_ptr += sizeof(EthernetHeader);

  IPHeader *ip = reinterpret_cast<IPHeader *>(data_ptr);
  std::memset(ip, 0, sizeof(IPHeader));
  ip->version_ihl = (4 << 4) | 5; // IPv4, IHL=5
  ip->protocol = 17;              // UDP
  // 10.0.<src_ip_octet3>.1, network byte order -- third octet carries the
  // synthetic FPGA id when the config overwrites it from the IP address.
  uint32_t host_ip = 0x0a000001;
  host_ip = (host_ip & ~0x0000ff00U) |
            (static_cast<uint32_t>(src_ip_octet3) << 8);
  ip->src_ip = htonl(host_ip);
  ip->dst_ip = htonl(0x0a000002);
  data_ptr += sizeof(IPHeader);

  UDPHeader *udp = reinterpret_cast<UDPHeader *>(data_ptr);
  std::memset(udp, 0, sizeof(UDPHeader));
  data_ptr += sizeof(UDPHeader);

  CustomHeader *custom = reinterpret_cast<CustomHeader *>(data_ptr);
  custom->sample_count = sample_count;
  custom->fpga_id = fpga_id;
  custom->freq_channel = freq_channel;
  std::memset(custom->padding, 0, sizeof(custom->padding));
  data_ptr += sizeof(CustomHeader);

  auto *payload = reinterpret_cast<typename T::PacketPayloadType *>(data_ptr);
  for (size_t r = 0; r < T::NR_RECEIVERS_PER_PACKET; ++r) {
    for (size_t p = 0; p < T::NR_POLARIZATIONS; ++p) {
      payload->scales[r][p] = scale_fn(static_cast<int>(r), static_cast<int>(p));
    }
  }
  for (size_t t = 0; t < T::NR_TIME_STEPS_PER_PACKET; ++t) {
    for (size_t r = 0; r < T::NR_RECEIVERS_PER_PACKET; ++r) {
      for (size_t p = 0; p < T::NR_POLARIZATIONS; ++p) {
        payload->data[t][r][p] = sample_fn(static_cast<int>(t), static_cast<int>(r),
                                            static_cast<int>(p));
      }
    }
  }

  return sizeof(EthernetHeader) + sizeof(IPHeader) + sizeof(UDPHeader) +
         sizeof(CustomHeader) + sizeof(typename T::PacketPayloadType);
}

// Convenience overload for the common case of a uniform sample value and a
// uniform scale across the whole packet (mirrors create_lambda_packet's `int
// val` shorthand).
template <typename T>
size_t build_constant_lambda_wire_packet(uint8_t *dest, uint64_t sample_count,
                                         uint32_t fpga_id, uint16_t freq_channel,
                                         std::complex<int8_t> sample_value,
                                         int16_t scale_value,
                                         uint8_t src_ip_octet3 = 0) {
  return build_lambda_wire_packet<T>(
      dest, sample_count, fpga_id, freq_channel,
      [&](int, int, int) { return sample_value; },
      [&](int, int) { return scale_value; }, src_ip_octet3);
}

// Builds one wire-format packet directly into a real ProcessorStateBase's
// current write slot and pushes it through the same
// get_current_write_pointer/add_received_packet_metadata/get_next_write_pointer
// sequence that live packet capture uses (see ProcessorStateTest::add_packet
// in test_processor.cu). This is the seam SyntheticPipelineRun drives.
template <typename T, typename SampleFn, typename ScaleFn>
void feed_lambda_packet(ProcessorStateBase &state, uint64_t sample_count,
                        uint32_t fpga_id, uint16_t freq_channel,
                        SampleFn &&sample_fn, ScaleFn &&scale_fn,
                        uint8_t src_ip_octet3 = 0) {
  uint8_t *data_ptr = reinterpret_cast<uint8_t *>(state.get_current_write_pointer());
  size_t total_length = build_lambda_wire_packet<T>(
      data_ptr, sample_count, fpga_id, freq_channel,
      std::forward<SampleFn>(sample_fn), std::forward<ScaleFn>(scale_fn),
      src_ip_octet3);

  struct sockaddr_in addr {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  state.add_received_packet_metadata(static_cast<int>(total_length), addr);
  state.get_next_write_pointer();
}

} // namespace test_support
