#include "spatial/ethernet.hpp"
#include "spatial/spatial.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <complex>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>

#define PORT 12345
#define BUFFER_SIZE 4096
#define RING_BUFFER_SIZE 1000
#define MIN_PCAP_HEADER_SIZE 64

constexpr int NUM_ITERATIONS = 10;
constexpr int NUM_FRAMES_PER_ITERATION = 100;
constexpr int NR_TOTAL_FRAMES_PER_CHANNEL = 5 * NUM_FRAMES_PER_ITERATION;
constexpr int NR_FPGA_SOURCES = 4;
constexpr int NR_RECEIVERS_PER_PACKET = NR_RECEIVERS / NR_FPGA_SOURCES;

int randomChannel() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, NR_CHANNELS - 1);
  return dist(gen);
}
int randomSource() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, NR_FPGA_SOURCES - 1);
  return dist(gen);
}

// typedef Sample Packet[NR_RECEIVERS_PER_PACKET][NR_POLARIZATIONS]
//                      [NR_TIME_STEPS_PER_PACKET];
typedef Sample Packet[NR_TIME_STEPS_PER_PACKET][NR_RECEIVERS_PER_PACKET];
typedef Sample PacketSamples[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION]
                            [NR_RECEIVERS][NR_POLARIZATIONS]
                            [NR_TIME_STEPS_PER_PACKET];
typedef bool SampleOccupancy[NR_CHANNELS][spatial::NR_BLOCKS_FOR_CORRELATION]
                            [NR_RECEIVERS];
struct PacketData {
  Packet data;
};

struct BufferState {
  bool is_ready;
  int start_seq;
  int end_seq;
  std::array<bool, NR_CHANNELS> is_populated{};
};

// Packet storage for ring buffer
struct PacketEntry {
  uint8_t data[BUFFER_SIZE];
  int length;
  struct sockaddr_in sender_addr;
  struct timeval timestamp;
  bool processed; // 0 = unprocessed, 1 = processed
};

typedef PacketEntry Packets[RING_BUFFER_SIZE];
// using Tin = int8_t;
// using Tscale = int16_t;
constexpr int NR_TIMES_PER_PACKET = 64;
// constexpr int NR_ACTUAL_RECEIVERS = 20;
constexpr int COMPLEX = 2;

using PacketDataStructure =
    std::complex<Tin>[NR_TIMES_PER_PACKET][NR_ACTUAL_RECEIVERS];

using PacketScaleStructure = Tscale[NR_ACTUAL_RECEIVERS];

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

struct ProcessorState {
  PacketSamples *d_samples[NR_BUFFERS];
  PacketEntry *d_packet_data[RING_BUFFER_SIZE];
  std::array<BufferState, NR_BUFFERS> buffers;
  uint64_t latest_packet_received[NR_CHANNELS][NR_FPGA_SOURCES] = {};
  // what is this one used for again?
  int current_buffer = 0;
  int write_index = 0;

  ProcessorState() {
    std::fill_n(d_samples, NR_BUFFERS, nullptr);
    std::fill_n(d_packet_data, RING_BUFFER_SIZE, nullptr);
    try {
      // This will eventually be replaced by cuda calls.
      for (auto i = 0; i < RING_BUFFER_SIZE; ++i) {
        d_packet_data[i] = (PacketEntry *)calloc(1, sizeof(PacketEntry));
        if (!d_packet_data[i]) {
          throw std::bad_alloc();
        }
      }

      for (auto i = 0; i < NR_BUFFERS; ++i) {
        d_samples[i] = (PacketSamples *)calloc(1, sizeof(PacketSamples));
        if (!d_samples[i]) {
          throw std::bad_alloc();
        }
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~ProcessorState() { cleanup(); }

  // removing copy / move possibilities.
  ProcessorState(const ProcessorState &) = delete;
  ProcessorState &operator=(const ProcessorState &) = delete;
  ProcessorState(const ProcessorState &&) = delete;
  ProcessorState &operator=(ProcessorState &&) = delete;

private:
  void cleanup() {

    // This will eventually be replaced by cuda calls.
    for (auto i = 0; i < RING_BUFFER_SIZE; ++i) {
      free(d_packet_data[i]);
    }

    for (auto i = 0; i < NR_BUFFERS; ++i) {
      if (d_samples[i]) {
        free(d_samples[i]);
        d_samples[i] = nullptr;
      }
    }
  }
};
// Global ring buffer
static std::mutex buffer_mutex;
static int running = 1;

// Statistics
static std::atomic<unsigned long long> packets_received = 0;
static std::atomic<unsigned long long> packets_processed = 0;

void get_next_write_index(ProcessorState &state) {
  state.write_index = (state.write_index + 1) % RING_BUFFER_SIZE;
  while (state.d_packet_data[state.write_index] == nullptr ||
         !state.d_packet_data[state.write_index]->processed) {
    state.write_index = (state.write_index + 1) % RING_BUFFER_SIZE;
  };
}

void store_packet(uint8_t *data, int length, struct sockaddr_in *sender,
                  ProcessorState &state) {
  std::lock_guard<std::mutex> lock(buffer_mutex);

  PacketEntry *entry = state.d_packet_data[state.write_index];
  memcpy(entry->data, data, length);
  entry->length = length;
  entry->sender_addr = *sender;
  gettimeofday(&entry->timestamp, NULL);
  entry->processed = 0;

  get_next_write_index(state);
  packets_received.fetch_add(1, std::memory_order_relaxed);
}

ProcessedPacket parse_custom_packet(PacketEntry *entry) {
  ProcessedPacket result = {0};

  if (entry->length < MIN_PCAP_HEADER_SIZE) {
    printf("Packet too small for custom headers\n");
    return result;
  }

  // Parse your custom packet structure
  const EthernetHeader *eth = (const EthernetHeader *)entry->data;
  if (ntohs(eth->ethertype) != 0x0800) {
    printf("Not IPv4 packet\n");
    return result;
  }

  const CustomHeader *custom = (const CustomHeader *)(entry->data + 42);

  result.sample_count = custom->sample_count;
  result.fpga_id = custom->fpga_id;
  result.freq_channel = custom->freq_channel;
  result.timestamp = entry->timestamp;

  // Point to payload (after headers)
  result.payload =
      reinterpret_cast<PacketPayload *>(entry->data + MIN_PCAP_HEADER_SIZE);
  result.payload_size = entry->length - MIN_PCAP_HEADER_SIZE;
  result.original_packet_processed = &entry->processed;

  return result;
}

void copy_data_to_input_buffer_if_able(ProcessedPacket &pkt,
                                       ProcessorState &state) {

  state.latest_packet_received[pkt.freq_channel][pkt.fpga_id] =
      std::max(state.latest_packet_received[pkt.freq_channel][pkt.fpga_id],
               pkt.sample_count);

  // copy to correct place or leave it.
  for (int buffer = 0; buffer < NR_BUFFERS; ++buffer) {
    int buffer_index = (state.current_buffer + buffer) % NR_BUFFERS;
    int packet_index = pkt.sample_count - state.buffers[buffer_index].start_seq;

    if (buffer == 0 && packet_index < 0) {
      // This means that this packet is less than the lowest possible
      // start token. Maybe an out-of-order packet that's coming in?
      // Regardless we can't do anything with this.
      std::cout << "Discarding packet as it is before current buffer "
                   "with begin_seq "
                << state.buffers[state.current_buffer].start_seq
                << " actually has packet_index " << pkt.sample_count
                << std::endl;
      *pkt.original_packet_processed = true;
      break;
    }

    if (packet_index >= 0 && packet_index < NR_PACKETS_FOR_CORRELATION) {
      int receiver_index = pkt.fpga_id * NR_RECEIVERS_PER_PACKET;
      std::cout << "Copying data to packet_index " << packet_index
                << " and receiver index " << receiver_index << " of buffer "
                << (state.current_buffer + buffer_index) % NR_BUFFERS
                << std::endl;
      std::memcpy(
          &(*state.d_samples[buffer_index])[pkt.freq_channel][packet_index]
                                           [receiver_index],
          // this is almost certainly not right.
          pkt.payload->data, sizeof(PacketDataStructure));
      *pkt.original_packet_processed = true;
      break;
    }
  }
}
void process_packet_data(PacketEntry *pkt, ProcessorState &state) {
  // This is where you'd do your actual processing
  // For now, just print the info and simulate some work

  ProcessedPacket parsed = parse_custom_packet(pkt);
  printf("Processing packet: sample_count=%lu, freq_channel=%u, fpga_id=%u, "
         "payload=%d bytes\n",
         parsed.sample_count, parsed.freq_channel, parsed.fpga_id,
         parsed.payload_size);

  printf("First data point...%i + %i i\n", parsed.payload->data[0][0].real(),
         parsed.payload->data[0][0].imag());
  // Simulate processing time
  copy_data_to_input_buffer_if_able(parsed, state);
  if (*parsed.original_packet_processed) {
    packets_processed.fetch_add(1, std::memory_order_relaxed);
  }
}

void advance_to_next_buffer(ProcessorState &state) {
  state.buffers[state.current_buffer].is_ready = false;

  state.buffers[state.current_buffer].is_ready = true;
  int old_buffer = state.current_buffer;
  int end_current_buffer_seq = state.buffers[old_buffer].end_seq;

  // Move to next buffer
  state.current_buffer = (state.current_buffer + 1) % NR_BUFFERS;

  std::cout << "current_buffer is " << state.current_buffer
            << " and is it ready? "
            << state.buffers[state.current_buffer].is_ready << std::endl;

  // this is kinda assuming there's an async callback that will
  // ready the buffer after the data has been transferred out.
  while (!state.buffers[state.current_buffer].is_ready) {
    std::cout << "Waiting for buffer to be ready..." << std::endl;
  }

  // Reset new current buffer
  std::memset(std::begin(state.buffers[state.current_buffer].is_populated),
              (int)false, NR_CHANNELS);
  state.buffers[state.current_buffer].start_seq = end_current_buffer_seq + 1;
  state.buffers[state.current_buffer].end_seq =
      state.buffers[state.current_buffer].start_seq +
      NR_PACKETS_FOR_CORRELATION - 1;

  // Update old buffer for future use
  int max_end_seq_in_buffers = 0;
  for (auto i = 0; i < NR_BUFFERS; ++i) {
    max_end_seq_in_buffers =
        std::max(max_end_seq_in_buffers, state.buffers[i].end_seq);
  }
  state.buffers[old_buffer].start_seq = max_end_seq_in_buffers + 1;
  state.buffers[old_buffer].end_seq =
      state.buffers[old_buffer].start_seq + NR_PACKETS_FOR_CORRELATION - 1;

  std::cout
      << "Current buffer is all complete. Moving to next buffer which is #"
      << state.current_buffer << std::endl;
  std::cout << "New buffer starts at packet "
            << state.buffers[state.current_buffer].start_seq << " and ends at "
            << state.buffers[state.current_buffer].end_seq << "\n";
}
// Receiver thread - continuously receives packets
void receiver_thread(int sockfd, ProcessorState &state) {
  uint8_t buffer[BUFFER_SIZE];
  struct sockaddr_in client_addr;
  socklen_t client_len = sizeof(client_addr);

  printf("Receiver thread started\n");

  while (running) {
    int received = recvfrom(sockfd, buffer, BUFFER_SIZE, 0,
                            (struct sockaddr *)&client_addr, &client_len);

    if (received < 0) {
      if (errno == EINTR)
        continue;
      perror("recvfrom");
      break;
    }

    // Store in ring buffer
    store_packet(buffer, received, &client_addr, state);
  }

  printf("Receiver thread exiting\n");
}

void check_buffer_completion(ProcessorState &state) {
  for (auto channel = 0; channel < NR_CHANNELS; ++channel) {
    std::cout << "Check if buffers are complete for channel " << channel
              << std::endl;

    if (std::all_of(std::begin(state.latest_packet_received[channel]),
                    std::end(state.latest_packet_received[channel]),
                    [&state](int x) {
                      return x >= state.buffers[state.current_buffer].end_seq;
                    })) {
      state.buffers[state.current_buffer].is_populated[channel] = true;
      std::cout << "Buffer is complete for channel " << channel << ".\n";
    } else {
      std::cout << "Buffer is not complete for channel " << channel
                << " as end_seq is "
                << state.buffers[state.current_buffer].end_seq
                << " and latest_packet_receives are ";
      for (int check = 0; check < NR_FPGA_SOURCES; ++check) {
        std::cout << state.latest_packet_received[channel][check] << ", ";
      }
      std::cout << std::endl;
    }
  }
}
// Processor thread - continuously processes packets
void processor_thread(ProcessorState &state) {
  printf("Processor thread started\n");

  while (running) {
    for (auto i = 0; i < RING_BUFFER_SIZE; ++i) {
      PacketEntry *entry = state.d_packet_data[i];

      if (entry->length == 0 || entry->processed == true) {
        continue;
      }

      process_packet_data(entry, state);

      check_buffer_completion(state);
      if (std::all_of(state.buffers[state.current_buffer].is_populated.begin(),
                      state.buffers[state.current_buffer].is_populated.end(),
                      [](bool i) { return i; })) {
        // Send off data to be processed by CUDA pipeline.
        // Then advance to next buffer and keep iterating.
        advance_to_next_buffer(state);
      }
    }
  }

  printf("Processor thread exiting\n");
}

void print_startup_info() {

  // Startup debug info
  std::cout << "NR_CHANNELS: " << NR_CHANNELS << std::endl;
  std::cout << "NUM_FRAMES_PER_ITERATION: " << NUM_FRAMES_PER_ITERATION
            << std::endl;
  std::cout << "NR_TOTAL_FRAMES_PER_CHANNEL: " << NR_TOTAL_FRAMES_PER_CHANNEL
            << std::endl;
  std::cout << "NR_FPGA_SOURCES: " << NR_FPGA_SOURCES << std::endl;
  std::cout << "NR_RECEIVERS: " << NR_RECEIVERS << std::endl;
  std::cout << "NR_RECEIVERS_PER_PACKET: " << NR_RECEIVERS_PER_PACKET
            << std::endl;
  std::cout << "NR_PACKETS_FOR_CORRELATION: " << NR_PACKETS_FOR_CORRELATION
            << std::endl;
}

void initialize_buffers(ProcessorState &state) {

  for (auto i = 0; i < NR_BUFFERS; ++i) {
    state.buffers[i].start_seq = i * NR_PACKETS_FOR_CORRELATION;
    state.buffers[i].end_seq = (i + 1) * NR_PACKETS_FOR_CORRELATION - 1;
    state.buffers[i].is_ready = true;
  }
}

int main() {
  print_startup_info();

  ProcessorState state;

  initialize_buffers(state);
  int sockfd;
  struct sockaddr_in server_addr;

  printf("UDP Server with concurrent processing starting on port %d...\n",
         PORT);
  printf("Ring buffer size: %d packets\n\n", RING_BUFFER_SIZE);
  printf("Size of PacketPayload is %lu bytes...\n", sizeof(PacketPayload));
  // Create UDP socket
  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0) {
    perror("socket");
    return 1;
  }

  // Allow address reuse
  int reuse = 1;
  if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
    perror("setsockopt");
  }

  // Setup server address
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = INADDR_ANY;
  server_addr.sin_port = htons(PORT);

  // Bind socket
  if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
    perror("bind");
    close(sockfd);
    return 1;
  }

  printf("Server listening on 0.0.0.0:%d\n", PORT);
  printf("Press Ctrl+C to stop\n\n");

  // Start receiver thread
  std::thread receiver(receiver_thread, sockfd, std::ref(state));

  // Start processor thread
  std::thread processor(processor_thread, std::ref(state));

  // Print statistics periodically
  while (running) {
    sleep(5);
    printf(
        "Stats: Received=%llu, Processed=%llu\n",
        (unsigned long long)packets_received.load(std::memory_order_relaxed),
        (unsigned long long)packets_processed.load(std::memory_order_relaxed));
  }

  // Cleanup
  printf("\nShutting down...\n");
  running = 0;
  receiver.join();
  processor.join();
  close(sockfd);
  return 0;
}
