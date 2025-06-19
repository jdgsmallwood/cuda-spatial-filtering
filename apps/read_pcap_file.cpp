#include "spatial/ethernet.hpp"
#include "spatial/spatial.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudawrappers/cu.hpp>
#include <functional>
#include <iostream>
#include <libtcc/Correlator.h>
#include <optional>
#include <pcap/pcap.h>
#include <stdexcept>
#include <vector>
constexpr int g_FILTERBANKS = 5;
constexpr int NR_BLOCKS_FOR_CORRELATION = 50;
constexpr int NUM_BUFFERS = 2;
constexpr int NR_ACTUAL_RECEIVERS = 20;
constexpr int NR_TIME_STEPS_PER_PACKET = 64;

typedef int8_t Tin;
typedef int16_t Tscale;

struct LambdaWrapper {
  // This exists to allow calling of std::memcpy on host into buffer in async
  // pipeline.
  std::function<void()> func;

  static void launch(void *data) {
    std::unique_ptr<LambdaWrapper> wrapper(static_cast<LambdaWrapper *>(data));
    wrapper->func();
  }
};

void process_packet(const u_char *packet, const int size,
                    std::vector<SampleFrame> &agg_samples,
                    const int start_seq_id, const int start_freq) {
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
      (custom->sample_count - start_seq_id) / NR_TIME_STEPS_PER_PACKET;
  const int packet_num_in_frame = packet_num % NR_BLOCKS_FOR_CORRELATION;
  const int sample_frame_to_populate = packet_num / NR_BLOCKS_FOR_CORRELATION;
  const int num_blocks_per_packet =
      NR_TIME_STEPS_PER_PACKET / NR_TIMES_PER_BLOCK; // 8
  printf("Packet number is %u. Seq is %u and start_seq was %u\n", packet_num,
         custom->sample_count, start_seq_id);
  printf("Will write to block %u in frame %u\n", packet_num_in_frame,
         sample_frame_to_populate);

  // Scale factors start at offset 64
  const Tscale *scales = reinterpret_cast<const Tscale *>(packet + 64);

  // Filterbank samples after scales
  const Tin *samples = reinterpret_cast<const Tin *>(
      packet + 64 + g_FILTERBANKS * sizeof(Tscale) * 4);
  const int freq_idx = custom->freq_channel - start_freq;
  for (auto k = 0; k < num_blocks_per_packet; ++k) {
    int pkt_idx = num_blocks_per_packet * packet_num_in_frame + k;
    for (auto i = 0; i < NR_TIMES_PER_BLOCK; ++i) {
      for (auto j = 0; j < NR_ACTUAL_RECEIVERS; ++j) {
        agg_samples[sample_frame_to_populate].data[freq_idx][pkt_idx][j][0][i] =
            Sample(scales[j] * samples[2 * NR_ACTUAL_RECEIVERS * i + 2 * j],
                   scales[j] *
                       samples[2 * NR_ACTUAL_RECEIVERS * i + 2 * j + 1]);
      }
    }
  }
}

int main(int argc, char *argv[]) {
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

  cudaStream_t streams[NUM_BUFFERS];
  cudaEvent_t input_transfer_done[NUM_BUFFERS];

  for (auto i = 0; i < NUM_BUFFERS; ++i) {
    cudaStreamCreate(&streams[i]);
    cudaEventCreate(&input_transfer_done[i]);
  }

  Samples *h_samples[NUM_BUFFERS];
  Visibilities *h_visibilities[NUM_BUFFERS];

  for (auto i = 0; i < NUM_BUFFERS; ++i) {
    cudaMallocHost(&h_samples[i], sizeof(Samples));
    cudaMallocHost(&h_visibilities[i], sizeof(Visibilities));
  }

  Samples *d_samples[NUM_BUFFERS];
  Visibilities *d_visibilities[NUM_BUFFERS];
  for (auto i = 0; i < NUM_BUFFERS; ++i) {
    cudaMalloc(&d_samples[i], sizeof(Samples));
    cudaMalloc(&d_visibilities[i], sizeof(Visibilities));
  }
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
  const int number_of_aggregated_packets =
      num_packets_captured / NR_BLOCKS_FOR_CORRELATION /
      (end_freq_channel - start_freq_channel + 1);
  printf("Storing in groups of 10. Number of aggregated packets is %u\n",
         number_of_aggregated_packets);
  if (res == -1) {
    printf("Error reading packets: %s\n", pcap_geterr(handle));
  }
  // second pass to actually get the data

  pcap_close(handle);
  // Reopen the file to return to the beginning of the file.
  handle = pcap_open_offline(argv[1], errbuf);
  std::vector<SampleFrame> aggregated_samples(number_of_aggregated_packets);
  std::vector<VisibilityFrame> aggregated_vis(number_of_aggregated_packets);
  printf("Setting aggregated_samples memory to zero\n");
  for (auto &frame : aggregated_samples) {
    std::memset(&frame, 0, sizeof(SampleFrame));
  }

  printf("test - see nonzero samples\n");
  print_nonzero_samples(&aggregated_samples[0].data);
  printf("Setting aggregated visibilities memory to zero\n");
  for (auto &frame : aggregated_vis) {
    std::memset(&frame, 0, sizeof(VisibilityFrame));
  }
  printf("test - see nonzero visibilities\n");
  print_nonzero_visibilities(&aggregated_vis[0].data);
  printf("Processing packets\n");
  // TODO: Will need to sort out handling of not full packet parcels.
  // will this just be fine and it will be handled by zero padding?
  while ((res = pcap_next_ex(handle, &header, &data)) >= 0) {
    if (res == 0)
      continue; // Timeout in live capture, ignore for offline
                //  printf("Timestamp: %u\n", header->ts);
                //  printf("Packet length: %u\n", header->len);
    process_packet(data, header->len, aggregated_samples, start_seq_num,
                   start_freq_channel);
  }
  pcap_close(handle);

  int current_buffer = 0;
  // std::atomic is overkill right now but if we end up using multi-threading at
  // some point this sidesteps a race condition.
  std::atomic<int> last_frame_processed = 0;
  bool processing = true;

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
  while (processing) {

    if (cudaEventQuery(input_transfer_done[current_buffer]) == cudaSuccess) {
      printf("Beginning new processing loop....\n");
      int next_frame_to_capture = last_frame_processed.fetch_add(1);
      printf("Next frame to capture is %u for stream %u\n",
             next_frame_to_capture, current_buffer);
      // Use this for the lambda function to capture current value.
      int buffer_value = current_buffer;
      if (next_frame_to_capture + 1 >= number_of_aggregated_packets) {
        processing = false;
        printf(
            "Finishing processing loop as next frame is %u which +1 is greater "
            "than or equal to the number of aggregated_packets %u\n",
            next_frame_to_capture, number_of_aggregated_packets);
      }

      std::memcpy(h_samples[current_buffer],
                  &aggregated_samples[next_frame_to_capture].data,
                  sizeof(Samples));

      cudaMemcpyAsync(d_samples[current_buffer], h_samples[current_buffer],
                      sizeof(Samples), cudaMemcpyHostToDevice,
                      streams[current_buffer]);
      // Now we can start preparing the next buffer for transport to the GPU.
      cudaEventRecord(input_transfer_done[current_buffer],
                      streams[current_buffer]);

      correlator.launchAsync((CUstream)streams[current_buffer],
                             (CUdeviceptr)d_visibilities[current_buffer],
                             (CUdeviceptr)d_samples[current_buffer]);
      checkCudaCall(
          cudaMemcpyAsync(h_visibilities[current_buffer],
                          d_visibilities[current_buffer], sizeof(Visibilities),
                          cudaMemcpyDeviceToHost, streams[current_buffer]));
      auto lambda = std::make_unique<LambdaWrapper>(
          LambdaWrapper{[next_frame_to_capture, buffer_value, &aggregated_vis,
                         &h_visibilities]() {
            std::memcpy(&aggregated_vis[next_frame_to_capture].data,
                        h_visibilities[buffer_value], sizeof(Visibilities));
          }});

      cudaLaunchHostFunc(streams[current_buffer], LambdaWrapper::launch,
                         lambda.release());
    }

    current_buffer = (current_buffer + 1) % NUM_BUFFERS;
  }
  printf("Synchronizing...\n");
  cudaDeviceSynchronize();

  printf("Starting to print visibilities...\n");

  for (auto i = 0; i < number_of_aggregated_packets; ++i) {
    printf("Visibilities for %u:\n", i);
    print_nonzero_visibilities(&aggregated_vis[i].data);
  }
  for (auto i = 0; i < NUM_BUFFERS; ++i) {
    cudaFreeHost(h_samples[i]);
    cudaFreeHost(h_visibilities[i]);
    cudaFree(d_samples[i]);
    cudaFree(d_visibilities[i]);
    cudaStreamDestroy(streams[i]);
    cudaEventDestroy(input_transfer_done[i]);
  }
  return 0;
}
