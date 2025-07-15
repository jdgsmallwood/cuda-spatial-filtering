#include "spatial/ethernet.hpp"
#include "spatial/spatial.cuh"
#include "spatial/spatial.hpp"
#include "spatial/tensor.hpp"
#include <algorithm>
#include <atomic>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudawrappers/cu.hpp>
#include <iostream>
#include <memory>
#include <pcap/pcap.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define DEBUG 1

PCAPInfo get_pcap_info(char *file_name) {

  char errbuf[PCAP_ERRBUF_SIZE];
  pcap_t *handle = pcap_open_offline(file_name, errbuf);
  if (!handle) {
    printf("pcap_open_offline failed: %s\n", errbuf);
  }

  struct pcap_pkthdr *header;
  const u_char *data;
  int res;

  // figure out what the data looks like.
  int start_freq_channel = -1;
  int end_freq_channel = -1;

  int start_seq_num = -1;
  int end_seq_num = -1;
  bool first_packet = true;

  int num_packets_captured = 0;
  while ((res = pcap_next_ex(handle, &header, &data)) >= 0) {
    if (res == 0)
      continue; // Timeout in live capture, ignore for offline
    num_packets_captured++;
    PacketInfo p = get_packet_info(data, header->len);

    if (first_packet) {
      start_seq_num = p.sample_count;
      end_seq_num = p.sample_count;

      start_freq_channel = p.freq_channel;
      end_freq_channel = p.freq_channel;
      first_packet = false;
    } else {
      if (p.sample_count < start_seq_num) {
        start_seq_num = p.sample_count;
      }

      if (p.sample_count > end_seq_num) {
        end_seq_num = p.sample_count;
      }

      if (p.freq_channel < start_freq_channel) {
        start_freq_channel = p.freq_channel;
      }

      if (p.freq_channel > end_freq_channel) {
        end_freq_channel = p.freq_channel;
      }
    }
  }

  PCAPInfo p = {start_freq_channel, end_freq_channel, start_seq_num,
                end_seq_num, num_packets_captured};
  if (res == -1) {
    printf("Error reading packets: %s\n", pcap_geterr(handle));
  }
  pcap_close(handle);
  return p;
}

int main(int argc, char *argv[]) {
  /* Read data from a PCAP file, run through the Tensor-Core Correlator and
   * output visibilities to stderr.
   *
   * Structure:
   * 1. Read through PCAP packets to get frequency channels, length of packets
   * etc.
   * 2. Read through again to store data.
   * 3. Load data to h_samples, h_visibilities
   * 4. Run CUDA operations & print
   *
   * We use a multiple buffer / stream system to allow stream concurrency on the
   * GPU.
   *
   * */

  // Check if any padding
  std::cout << "sizeof(__half): " << sizeof(__half) << std::endl;
  std::cout << "sizeof(std::complex<__half>): " << sizeof(std::complex<__half>)
            << std::endl;
  static_assert(sizeof(std::complex<__half>) == 2 * sizeof(__half),
                "Padding detected in std::complex<__half>");

  std::cout << "sizeof(float): " << sizeof(float) << std::endl;
  std::cout << "sizeof(std::complex<float>): " << sizeof(std::complex<float>)
            << std::endl;
  static_assert(sizeof(std::complex<float>) == 2 * sizeof(float),
                "Padding detected in std::complex<float>");

  /*
   * PCAP Formatting.
   * */

  if (argc < 2) {
    printf("Usage: %s <pcap file>\n", argv[0]);
    return 1;
  }

  PCAPInfo info = get_pcap_info(argv[1]);
  printf("Start freq channel is %u and end freq channel is %u\n",
         info.start_freq, info.end_freq);
  printf("Start seq num is %u and end seq num is %u\n", info.start_seq,
         info.end_seq);

  printf("Total packets captured is %u\n", info.num_packets_captured);
  const int num_freq = info.end_freq - info.start_freq + 1;
  const int number_of_aggregated_packets = std::max(
      info.num_packets_captured / NR_PACKETS_FOR_CORRELATION / num_freq + 1, 1);
  printf("Storing in groups of 10. Number of aggregated packets is %u\n",
         number_of_aggregated_packets);
  // second pass to actually get the data

  // Reopen the file to return to the beginning of the file.
  char errbuf[PCAP_ERRBUF_SIZE];
  pcap_t *handle = pcap_open_offline(argv[1], errbuf);

  struct pcap_pkthdr *header;
  const u_char *data;
  int res;
  /*
   * PCAP Data Reading
   */

  // allocate pinned host memory
  Samples *h_samples;
  Visibilities *h_visibilities;
  std::complex<__half> *h_weights;
  BeamformedData *h_beamformed_data;

  constexpr int num_weights =
      NR_BEAMS * NR_RECEIVERS * NR_POLARIZATIONS * NR_CHANNELS;

  constexpr int num_eigen = NR_RECEIVERS * NR_CHANNELS * NR_POLARIZATIONS;

  cudaMallocHost(&h_samples, number_of_aggregated_packets * sizeof(Samples));
  cudaMallocHost(&h_visibilities,
                 number_of_aggregated_packets * sizeof(Visibilities));
  cudaMallocHost(&h_weights, num_weights * sizeof(std::complex<__half>));
  // not sure about this memory allocation.
  cudaMallocHost(&h_beamformed_data,
                 number_of_aggregated_packets * sizeof(BeamformedData));

  for (auto i = 0; i < num_weights; ++i) {
    h_weights[i] = std::complex<__half>(__float2half(1.0f), __float2half(1.0f));
  }

  std::vector<Tscale> scales(NR_ACTUAL_RECEIVERS);

  Tscale *h_scales;
  cudaMallocHost(&h_scales, NR_ACTUAL_BASELINES * sizeof(Tscale));

  printf("Setting h_samples & h_visibilities memory to zero\n");
  std::memset(h_samples, 0, number_of_aggregated_packets * sizeof(Samples));
  std::memset(h_visibilities, 0,
              number_of_aggregated_packets * sizeof(Visibilities));
  printf("Processing packets\n");
  while ((res = pcap_next_ex(handle, &header, &data)) >= 0) {
    if (res == 0)
      continue; // Timeout in live capture, ignore for offline
    process_packet(data, header->len, h_samples, scales, info.start_seq,
                   info.start_freq, NR_TIME_STEPS_PER_PACKET,
                   NR_PACKETS_FOR_CORRELATION, NR_TIMES_PER_BLOCK,
                   NR_ACTUAL_RECEIVERS);
  }
  pcap_close(handle);

  printf("h_scales:\n");
  for (auto i = 0; i < NR_ACTUAL_RECEIVERS; ++i) {
    for (auto j = 0; j <= i; ++j) {
      int baseline = i * (i + 1) / 2 + j;
      h_scales[baseline] = scales[i] * scales[j];
      printf("baseline (%i, %i, %i): %u x %u = %u\n", i, j, baseline, scales[i],
             scales[j], h_scales[baseline]);
    }
  }

  beamform((std::complex<int8_t> *)h_samples, (std::complex<__half> *)h_weights,
           (std::complex<float> *)h_beamformed_data,
           (std::complex<float> *)h_visibilities, number_of_aggregated_packets);

  /*
   * Output
   * */

  printf("Starting to print visibilities...\n");
  for (auto i = 0; i < number_of_aggregated_packets; ++i) {
    printf("Visibilities for %u:\n", i);
    print_nonzero_visibilities(&h_visibilities[i], h_scales);

    printf("Beams for %u:\n", i);
    print_nonzero_beams(&h_beamformed_data[i], NR_CHANNELS, NR_POLARIZATIONS,
                        NR_BEAMS,
                        NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK);
  }

  return 0;
}
