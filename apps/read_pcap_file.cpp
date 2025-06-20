#include "spatial/ethernet.hpp"
#include "spatial/spatial.hpp"
#include <algorithm>
#include <atomic>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudawrappers/cu.hpp>
#include <iostream>
#include <libtcc/Correlator.h>
#include <pcap/pcap.h>
#include <stdexcept>
#include <vector>
constexpr int NR_BLOCKS_FOR_CORRELATION = 50;
constexpr int NUM_BUFFERS = 2;
constexpr int NR_ACTUAL_RECEIVERS = 20;
constexpr int NR_TIME_STEPS_PER_PACKET = 64;

constexpr int NR_ACTUAL_BASELINES =
    NR_ACTUAL_RECEIVERS * (NR_ACTUAL_RECEIVERS + 1) / 2;
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

  /*
   * PCAP Formatting.
   * */

  if (argc < 2) {
    printf("Usage: %s <pcap file>\n", argv[0]);
    return 1;
  }

  char errbuf[PCAP_ERRBUF_SIZE];
  pcap_t *handle = pcap_open_offline(argv[1], errbuf);
  if (!handle) {
    printf("pcap_open_offline failed: %s\n", errbuf);
    return 1;
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

  printf("Start freq channel is %u and end freq channel is %u\n",
         start_freq_channel, end_freq_channel);
  printf("Start seq num is %u and end seq num is %u\n", start_seq_num,
         end_seq_num);

  printf("Total packets captured is %u\n", num_packets_captured);
  const int num_freq = end_freq_channel - start_freq_channel + 1;
  const int number_of_aggregated_packets =
      std::max(num_packets_captured / NR_BLOCKS_FOR_CORRELATION / num_freq, 1);
  printf("Storing in groups of 10. Number of aggregated packets is %u\n",
         number_of_aggregated_packets);
  if (res == -1) {
    printf("Error reading packets: %s\n", pcap_geterr(handle));
  }
  // second pass to actually get the data

  pcap_close(handle);
  // Reopen the file to return to the beginning of the file.
  handle = pcap_open_offline(argv[1], errbuf);

  /*
   * PCAP Data Reading
   * */

  // allocate pinned host memory
  Samples *h_samples;
  Visibilities *h_visibilities;

  cudaMallocHost(&h_samples, number_of_aggregated_packets * sizeof(Samples));
  cudaMallocHost(&h_visibilities,
                 number_of_aggregated_packets * sizeof(Visibilities));

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
                //  printf("Timestamp: %u\n", header->ts);
                //  printf("Packet length: %u\n", header->len);
    process_packet(data, header->len, h_samples, scales, start_seq_num,
                   start_freq_channel, NR_TIME_STEPS_PER_PACKET,
                   NR_BLOCKS_FOR_CORRELATION, NR_TIMES_PER_BLOCK,
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

  // create CUDA streams
  cudaStream_t streams[NUM_BUFFERS];
  cudaEvent_t input_transfer_done[NUM_BUFFERS];

  for (auto i = 0; i < NUM_BUFFERS; ++i) {
    cudaStreamCreate(&streams[i]);
    cudaEventCreate(&input_transfer_done[i]);
  }

  // create device pointers
  Samples *d_samples[NUM_BUFFERS];
  Visibilities *d_visibilities[NUM_BUFFERS];
  for (auto i = 0; i < NUM_BUFFERS; ++i) {
    cudaMalloc(&d_samples[i], sizeof(Samples));
    cudaMalloc(&d_visibilities[i], sizeof(Visibilities));
  }
  // start with these events in done state.
  for (auto i = 0; i < NUM_BUFFERS; ++i) {
    cudaEventRecord(input_transfer_done[i], streams[i]);
  }

  printf("Initializing correlator\n");
  tcc::Correlator correlator(cu::Device(0), inputFormat, NR_RECEIVERS,
                             NR_CHANNELS, NR_SAMPLES_PER_CHANNEL,
                             NR_POLARIZATIONS, NR_RECEIVERS_PER_BLOCK);
  printf("NR_RECEIVERS: %u\n", NR_RECEIVERS);
  printf("NR_CHANNELS: %u\n", NR_CHANNELS);
  printf("NR_SAMPLES_PER_CHANNEL: %u\n", NR_SAMPLES_PER_CHANNEL);
  printf("NR_TIMES_PER_BLOCK: %u\n", NR_TIMES_PER_BLOCK);
  printf("NR_POLARIZATIONS: %u\n", NR_POLARIZATIONS);
  printf("NR_RECEIVERS_PER_BLOCK: %u\n", NR_RECEIVERS_PER_BLOCK);
  printf("NR_ACTUAL_RECEIVERS: %u\n", NR_ACTUAL_RECEIVERS);
  printf("NR_BITS: %u\n", NR_BITS);
  printf("Launching processing loop...\n");
  int current_buffer = 0;
  // std::atomic is overkill right now but if we end up using multi-threading at
  // some point this sidesteps a race condition.
  std::atomic<int> last_frame_processed = 0;
  bool processing = true;

  // Main processing loop.
  while (processing) {
    if (cudaEventQuery(input_transfer_done[current_buffer]) == cudaSuccess) {
      printf("Beginning new processing loop....\n");
      int next_frame_to_capture = last_frame_processed.fetch_add(1);
      printf("Next frame to capture is %u for stream %u\n",
             next_frame_to_capture, current_buffer);
      // Use this for the lambda function to capture current value.
      if (next_frame_to_capture + 1 >= number_of_aggregated_packets) {
        processing = false;
        printf(
            "Finishing processing loop as next frame is %u which +1 is greater "
            "than or equal to the number of aggregated_packets %u\n",
            next_frame_to_capture, number_of_aggregated_packets);
      }

      cudaMemcpyAsync(d_samples[current_buffer],
                      h_samples[next_frame_to_capture], sizeof(Samples),
                      cudaMemcpyHostToDevice, streams[current_buffer]);
      // Now we can start preparing the next buffer for transport to the GPU.
      cudaEventRecord(input_transfer_done[current_buffer],
                      streams[current_buffer]);

      correlator.launchAsync((CUstream)streams[current_buffer],
                             (CUdeviceptr)d_visibilities[current_buffer],
                             (CUdeviceptr)d_samples[current_buffer]);
      checkCudaCall(
          cudaMemcpyAsync(h_visibilities[next_frame_to_capture],
                          d_visibilities[current_buffer], sizeof(Visibilities),
                          cudaMemcpyDeviceToHost, streams[current_buffer]));
    }

    current_buffer = (current_buffer + 1) % NUM_BUFFERS;
  }
  printf("Synchronizing...\n");
  cudaDeviceSynchronize();

  /*
   * Output
   * */

  printf("Starting to print visibilities...\n");
  for (auto i = 0; i < number_of_aggregated_packets; ++i) {
    printf("Visibilities for %u:\n", i);
    print_nonzero_visibilities(&h_visibilities[i], h_scales);
  }
  for (auto i = 0; i < NUM_BUFFERS; ++i) {
    cudaFree(d_samples[i]);
    cudaFree(d_visibilities[i]);
    cudaStreamDestroy(streams[i]);
    cudaEventDestroy(input_transfer_done[i]);
  }
  cudaFreeHost(h_samples);
  cudaFreeHost(h_visibilities);
  return 0;
}
