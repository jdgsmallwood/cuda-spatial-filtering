#pragma once
#include "spatial/logging.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline_base.hpp"
#include <complex>
#include <cuda.h>
#include <iostream>
#include <libtcc/Correlator.h>
#include <netinet/in.h>
// #include <sys/socket.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <pcap/pcap.h>
#include <queue>
#include <sys/time.h>

#define MIN_PCAP_HEADER_SIZE 64
#define BUFFER_SIZE 4096
#include <cuda_fp16.h>

// template <typename T>
// void eigendecomposition(float *h_eigenvalues, int n, const std::vector<T>
// *A); template <typename T> void d_eigendecomposition(float *d_eigenvalues,
// const int n,
//                           const int num_channels, const int
//                           num_polarizations, T *d_A, cudaStream_t stream);
void ccglib_mma(__half *A, __half *B, float *C, const int n_row,
                const int n_col, const int batch_size, int n_inner = -1);

void ccglib_mma_opt(__half *A, __half *B, float *C, const int n_row,
                    const int n_col, const int batch_size, int n_inner,
                    const int tile_size_x, const int tile_size_y);
template <size_t NR_CHANNELS> struct BufferState {
  bool is_ready;
  int start_seq;
  int end_seq;
  std::array<bool, NR_CHANNELS> is_populated{};
};

// forward declaration of GPUPipeline.
class GPUPipeline;

class ProcessorStateBase {
public:
  int current_buffer = 0;
  std::atomic<int> write_index = 0;
  std::atomic<int> read_index = 0;
  std::vector<uint32_t> fpga_ids{};
  bool buffers_initialized = false;
  int running = 1;
  unsigned long long packets_received = 0;
  unsigned long long packets_processed = 0;
  virtual void *get_next_write_pointer() = 0;
  virtual void *get_current_write_pointer() = 0;
  virtual void add_received_packet_metadata(const int length,
                                            const sockaddr_in &client_addr) = 0;
  virtual void release_buffer(const int buffer_index) = 0;
  virtual void set_pipeline(GPUPipeline *pipeline) = 0;
};
template <typename T, size_t NR_INPUT_BUFFERS = 2,
          size_t RING_BUFFER_SIZE = 1000>
class ProcessorState : public ProcessorStateBase {
public:
  typename T::PacketFinalDataType *d_samples[NR_INPUT_BUFFERS];
  typename T::PacketEntryType *d_packet_data[RING_BUFFER_SIZE];
  size_t MIN_FREQ_CHANNEL;
  size_t NR_BETWEEN_SAMPLES;
  size_t NR_PACKETS_FOR_CORRELATION;

  std::array<BufferState<T::NR_CHANNELS>, NR_INPUT_BUFFERS> buffers;
  uint64_t latest_packet_received[T::NR_CHANNELS][T::NR_FPGA_SOURCES] = {};
  mutable std::mutex buffer_index_mutex;
  GPUPipeline *pipeline_;
  // Constructor / Destructor
  ProcessorState(size_t nr_packets_for_correlation, size_t nr_between_samples,
                 size_t min_freq_channel)
      : NR_PACKETS_FOR_CORRELATION(nr_packets_for_correlation),
        NR_BETWEEN_SAMPLES(nr_between_samples),
        MIN_FREQ_CHANNEL(min_freq_channel) {
    std::fill_n(d_samples, NR_INPUT_BUFFERS, nullptr);
    std::fill_n(d_packet_data, RING_BUFFER_SIZE, nullptr);
    std::fill(modified_since_last_completion_check.begin(),
              modified_since_last_completion_check.end(), false);
    try {
      for (auto i = 0; i < RING_BUFFER_SIZE; ++i) {
        d_packet_data[i] = new typename T::PacketEntryType();
        if (!d_packet_data[i])
          throw std::bad_alloc();
        d_packet_data[i]->processed = true;
      }

      for (auto i = 0; i < NR_INPUT_BUFFERS; ++i) {
        d_samples[i] = new typename T::PacketFinalDataType();
        d_samples[i]->buffer_index = i;
        if (!d_samples[i])
          throw std::bad_alloc();
      }
    } catch (...) {
      cleanup();
      throw;
    }
  };
  ~ProcessorState() { cleanup(); };
  ProcessorState(const ProcessorState &) = delete;
  ProcessorState &operator=(const ProcessorState &) = delete;

  ProcessorState(ProcessorState &&) = delete;
  ProcessorState &operator=(ProcessorState &&) = delete;
  void set_pipeline(GPUPipeline *pipeline) { pipeline_ = pipeline; };
  bool get_next_write_index() {
    int next_write_index = -1;
    bool first_loop = true;
    while (next_write_index < 0 ||
           !d_packet_data[next_write_index]->processed) {
      // Get the next write index that has already been processed. This avoids
      // overwriting packets that have been left behind because their buffer was
      // not yet available.
      if (first_loop) {
        next_write_index = (write_index.load(std::memory_order_relaxed) + 1) %
                           RING_BUFFER_SIZE;
        first_loop = false;
      } else {
        next_write_index = (next_write_index + 1) % RING_BUFFER_SIZE;
      }
      if (next_write_index == read_index.load(std::memory_order_acquire)) {
        LOG_INFO("Ring buffer is full!! Dropping packets...");
        return false;
      }
    }
    write_index.store(next_write_index, std::memory_order_release);
    LOG_INFO("Next write index is...{}", next_write_index);
    return true;
  };
  void copy_data_to_input_buffer_if_able(
      ProcessedPacket<typename T::PacketScaleStructure,
                      typename T::PacketDataStructure> &pkt) {

    auto it = std::find(fpga_ids.begin(), fpga_ids.end(), pkt.fpga_id);
    size_t fpga_index;
    if (it != fpga_ids.end()) {
      fpga_index = std::distance(fpga_ids.begin(), it);
    } else {
      fpga_ids.push_back(pkt.fpga_id);
      fpga_index = fpga_ids.size() - 1;
    }

    int freq_channel = pkt.freq_channel - MIN_FREQ_CHANNEL;

    latest_packet_received[freq_channel][fpga_index] = std::max(
        latest_packet_received[freq_channel][fpga_index], pkt.sample_count);

    // copy to correct place or leave it.
    for (int buffer = 0; buffer < NR_INPUT_BUFFERS; ++buffer) {
      int buffer_index = (current_buffer + buffer) % NR_INPUT_BUFFERS;
      int packet_index = (pkt.sample_count - buffers[buffer_index].start_seq) /
                         NR_BETWEEN_SAMPLES;

      if (buffer == 0 && packet_index < 0) {
        // This means that this packet is less than the lowest possible
        // start token. Maybe an out-of-order packet that's coming in?
        // Regardless we can't do anything with this.
        LOG_INFO("Discarding packet as it is before current buffer with "
                 "begin_seq {} actually has packet_index {}",
                 buffers[current_buffer].start_seq, pkt.sample_count);
        *pkt.original_packet_processed = true;
        return;
      }

      if (packet_index >= 0 && packet_index < NR_PACKETS_FOR_CORRELATION) {
        int receiver_index = fpga_index * T::NR_RECEIVERS_PER_PACKET;
        LOG_DEBUG("Copying data to packet_index {} and channel index {} and "
                  "receiver_index {} of buffer {}",
                  packet_index, freq_channel, receiver_index,
                  (current_buffer + buffer_index) % NR_INPUT_BUFFERS);
        std::memcpy(
            &(*(*d_samples[buffer_index])
                   .samples)[freq_channel][packet_index][receiver_index],
            pkt.payload->data, sizeof(typename T::PacketDataStructure));
        std::memcpy(&(*(*d_samples[buffer_index])
                           .scales)[freq_channel][packet_index][receiver_index],
                    pkt.payload->scales,
                    sizeof(typename T::PacketScaleStructure));
        d_samples[buffer_index]
            ->arrivals[0][freq_channel][packet_index][fpga_index] = true;
        LOG_DEBUG("Setting original_packet_processed as true...");
        LOG_DEBUG("original_packet_processed_before={}",
                  *pkt.original_packet_processed);
        *(pkt.original_packet_processed) = true;
        LOG_DEBUG("DEBUG: original_packet_processed_after={}",
                  *pkt.original_packet_processed);

        return;
      }
    }
    int current_read_index = read_index.load(std::memory_order_relaxed);
    LOG_INFO("Packet with seq number {} and read index {} was unable to find a "
             "home. Adding to "
             "future_packet_queue...",
             pkt.sample_count, current_read_index);
    future_packet_queue.push(current_read_index);
  };
  void initialize_buffers(const int first_count) {

    for (auto i = 0; i < NR_INPUT_BUFFERS; ++i) {
      buffers[i].start_seq =
          first_count + i * NR_PACKETS_FOR_CORRELATION * NR_BETWEEN_SAMPLES;
      buffers[i].end_seq =
          first_count +
          ((i + 1) * NR_PACKETS_FOR_CORRELATION - 1) * NR_BETWEEN_SAMPLES;
      buffers[i].is_ready = true;

      // we know 0 will be the first one - so no need to add zero
      if (i != 0) {
        buffer_ordering_queue.push({i, buffers[i].start_seq});
      };
    }
  };
  void process_packet_data(typename T::PacketEntryType *pkt) {

    // This is where you'd do your actual processing
    // For now, just print the info and simulate some work

    ProcessedPacket parsed = pkt->parse();

    if (pkt->processed) {
      return;
    }
    LOG_INFO("Processing packet: sample_count={}, freq_channel={}, fpga_id={}, "
             "payload={} bytes",
             parsed.sample_count, parsed.freq_channel, parsed.fpga_id,
             parsed.payload_size);

    LOG_INFO("First data point...{} + {} i",
             parsed.payload->data[0][0][0].real(),
             parsed.payload->data[0][0][0].imag());

    if (!buffers_initialized) {
      LOG_INFO("Initializing buffers as this is the first packet...");
      initialize_buffers(parsed.sample_count);
      buffers_initialized = true;
    }
    // Simulate processing time
    copy_data_to_input_buffer_if_able(parsed);
    if (*parsed.original_packet_processed) {
      packets_processed += 1;
      modified_since_last_completion_check[parsed.freq_channel -
                                           MIN_FREQ_CHANNEL] = true;
    }
  };

  void execute_processing_pipeline_on_buffer(const int buffer_index) {};

  void release_buffer(const int buffer_index) {
    // This is called to let the processor know that the buffer has been
    // copied to the GPU and now can be overwritten.
    int max_end_seq_in_buffers = 0;
    // This is necessary to avoid multiple GPU threads competing / racing.
    std::lock_guard<std::mutex> lock(buffer_index_mutex);
    for (auto i = 0; i < NR_INPUT_BUFFERS; ++i) {
      max_end_seq_in_buffers =
          std::max(max_end_seq_in_buffers, buffers[i].end_seq);
    }
    buffers[buffer_index].start_seq =
        max_end_seq_in_buffers + 1 * NR_BETWEEN_SAMPLES;
    buffers[buffer_index].end_seq =
        buffers[buffer_index].start_seq +
        (NR_PACKETS_FOR_CORRELATION - 1) * NR_BETWEEN_SAMPLES;
    std::fill_n((bool *)d_samples[buffer_index]->arrivals,
                T::NR_CHANNELS * T::NR_PACKETS_FOR_CORRELATION *
                    T::NR_FPGA_SOURCES,
                false);
    buffers[buffer_index].is_ready = true;
    buffer_ordering_queue.push({buffer_index, buffers[buffer_index].start_seq});
  };

  void advance_to_next_buffer(const int current_buffer_start_seq) {

    // Move to next buffer
    // This is not necessarily the next buffer as the buffers can be
    // updated in any order. This will be the buffer that has the next
    // highest start_seq.
    LOG_INFO("advancing to next buffer...");
    {
      while (buffer_ordering_queue.empty()) {
        LOG_INFO("waiting for buffer to become available...");
      }
      std::lock_guard<std::mutex> lock(buffer_index_mutex);
      BufferOrder b = buffer_ordering_queue.top();
      buffer_ordering_queue.pop();
      LOG_INFO("Current buffer start seq is {}", current_buffer_start_seq);
      current_buffer = b.index;
    }

    LOG_INFO(
        "next current_buffer is {} and has start_seq {} and is it ready? {}",
        current_buffer, buffers[current_buffer].start_seq,
        buffers[current_buffer].is_ready);

    // this is kinda assuming there's an async callback that will
    // ready the buffer after the data has been transferred out.
    // This async callback will come from the GPUPipeline class.
    while (!buffers[current_buffer].is_ready) {
      LOG_INFO("Waiting for buffer to be ready...");
    }

    // Reset new current buffer
    // This is just NR_CHANNELS_DEF booleans that tell us if the current
    // buffer has all the data it needs.
    std::memset(std::begin(buffers[current_buffer].is_populated), (int)false,
                T::NR_CHANNELS);
    LOG_INFO(
        "Current buffer is all complete. Moving to next buffer which is #{}",
        current_buffer);
    LOG_INFO("New buffer starts at packet {} and ends at {}",
             buffers[current_buffer].start_seq,
             buffers[current_buffer].end_seq);
  }

  void check_buffer_completion() {

    if (!buffers_initialized) {
      return;
    }

    for (auto channel = 0; channel < T::NR_CHANNELS; ++channel) {
      if (buffers[current_buffer].is_populated[channel] ||
          !modified_since_last_completion_check[channel]) {
        continue;
      }
      LOG_INFO("Check if buffers are complete for channel {}", channel);

      if (std::all_of(std::begin(latest_packet_received[channel]),
                      std::end(latest_packet_received[channel]), [this](int x) {
                        return x >= buffers[current_buffer].end_seq;
                      })) {
        buffers[current_buffer].is_populated[channel] = true;
        LOG_INFO("Buffer is complete for channel {}", channel);
      } else {
        LOG_INFO("Buffer is not complete for channel {} as end_seq is {} and "
                 "latest_packet_receives are:",
                 channel, buffers[current_buffer].end_seq);
        for (int check = 0; check < T::NR_FPGA_SOURCES; ++check) {
          LOG_INFO("FPGA ID {} / Channel {}: {},", check, channel,
                   latest_packet_received[channel][check]);
        }
      }
      modified_since_last_completion_check[channel] = false;
    }
  };

  void process_packets() {

    using clock = std::chrono::high_resolution_clock;
    auto cpu_start = clock::now();
    auto cpu_end = clock::now();
    LOG_INFO("Processor thread started");
    int current_read_index;
    bool from_queue = false;
    while (running) {
      current_read_index = -1;
      from_queue = false;

      while (true) {
        current_read_index = read_index.load(std::memory_order_relaxed);

        if (current_read_index != write_index.load(std::memory_order_acquire)) {
          break;
        }
        if (future_packet_queue.size()) {
          current_read_index = future_packet_queue.front();
          LOG_INFO("Reading from future packet queue...position {}",
                   current_read_index);
          from_queue = true;
          future_packet_queue.pop();
          break;
        }

        if (!running) {
          // Will want to update this to finish buffer that's been processed.
          return;
        }
      }
      typename T::PacketEntryType *entry = d_packet_data[current_read_index];

      if (entry->length == 0 || entry->processed == true) {
        // if we don't increment the read index here it can get stuck!
        int new_read_index = (current_read_index + 1) % RING_BUFFER_SIZE;
        read_index.store(new_read_index, std::memory_order_release);
        LOG_INFO("New read index is {}", new_read_index);
        continue;
      }

      process_packet_data(entry);
      if (!from_queue) {
        int new_read_index = (current_read_index + 1) % RING_BUFFER_SIZE;
        read_index.store(new_read_index, std::memory_order_release);
        LOG_INFO("New read index is {}", new_read_index);
      }
      cpu_start = clock::now();
      handle_buffer_completion();
      cpu_end = clock::now();

      LOG_DEBUG("CPU time for buffer completion check: {} us",
                std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                      cpu_start)
                    .count());
    }

    LOG_INFO("Processor thread exiting");
  };

  void handle_buffer_completion() {
    using clock = std::chrono::high_resolution_clock;
    auto cpu_start = clock::now();
    auto cpu_end = clock::now();
    LOG_INFO("Checking buffer completion...");
    check_buffer_completion();
    if (std::all_of(buffers[current_buffer].is_populated.begin(),
                    buffers[current_buffer].is_populated.end(),
                    [](bool i) { return i; })) {
      LOG_INFO("Buffer is complete - passing to output pipeline...");
      // Send off data to be processed by CUDA pipeline.
      // Then advance to next buffer and keep iterating.
      if (pipeline_ == nullptr) {
        throw std::logic_error(
            "Pipeline has not been set. Ensure that set_pipeline has been "
            "called on ProcessorState class.");
      }

      buffers[current_buffer].is_ready = false;
      cpu_start = clock::now();
      LOG_INFO("Zeroing missing packets...");
      d_samples[current_buffer]->zero_missing_packets();
      cpu_end = clock::now();
      LOG_DEBUG("CPU time for zeroing packets: {} us",
                std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                      cpu_start)
                    .count());
      // order here is important. As the pipeline can async update
      // the start/end seqs we need to capture the
      // start_seq, then execute the pipeline, then advance using
      // the saved copy of the start_seq.
      const int current_buffer_start_seq = buffers[current_buffer].start_seq;
      LOG_INFO("Enqueueing buffer {} for pipeline...", current_buffer);
      buffers_ready_for_pipeline.push(current_buffer);
      LOG_INFO("Advancing to next buffer...");
      cpu_start = clock::now();
      advance_to_next_buffer(current_buffer_start_seq);
      cpu_end = clock::now();
      LOG_DEBUG("CPU time for advancing to next buffer: {} us",
                std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                      cpu_start)
                    .count());
      LOG_INFO("Done!");
    }
  };

  void pipeline_feeder() {
    LOG_INFO("Pipeline feeder starting up...");
    while (running) {
      if (buffers_ready_for_pipeline.size()) {
        size_t buffer_index = buffers_ready_for_pipeline.front();
        LOG_INFO("Buffer index {} picked up by pipeline feeder...",
                 buffer_index);
        pipeline_->execute_pipeline(d_samples[buffer_index]);
        buffers_ready_for_pipeline.pop();
      }
    }
    LOG_INFO("Pipeline feeder exiting!");
  }

  void *get_current_write_pointer() {
    return (void *)&(d_packet_data[write_index]->data);
  }
  void *get_next_write_pointer() {
    while (!get_next_write_index()) {
      LOG_DEBUG("Waiting for next pointer....");
    };
    return get_current_write_pointer();
  }

  void add_received_packet_metadata(const int length,
                                    const sockaddr_in &client_addr) {

    d_packet_data[write_index]->length = length;
    d_packet_data[write_index]->sender_addr = client_addr;
    d_packet_data[write_index]->processed = false;
    gettimeofday(&d_packet_data[write_index]->timestamp, NULL);
  }

private:
  void cleanup() {

    for (auto i = 0; i < RING_BUFFER_SIZE; ++i) {
      delete d_packet_data[i];
      d_packet_data[i] = nullptr;
    }

    for (auto i = 0; i < NR_INPUT_BUFFERS; ++i) {
      delete d_samples[i];
      d_samples[i] = nullptr;
    }
  };

  struct BufferOrder {
    int index;
    int start_seq;

    // Compare to make the priority queue a min-heap based on start_seq
    bool operator>(const BufferOrder &other) const {
      return start_seq > other.start_seq;
    }
  };

  std::queue<size_t> buffers_ready_for_pipeline;
  // This is for packets that arrive but their buffer is not yet created.
  // The read pointer moves on but keep these as things to process.
  // They will not be overwritten by the write pointer as it checks whether or
  // not they have been processed.
  std::queue<size_t> future_packet_queue;
  std::array<bool, T::NR_CHANNELS> modified_since_last_completion_check;
  std::priority_queue<BufferOrder, std::vector<BufferOrder>,
                      std::greater<BufferOrder>>
      buffer_ordering_queue;
};

class PacketInput {
public:
  virtual void get_packets(ProcessorStateBase &state) = 0;

  virtual ~PacketInput() = default;
};

class KernelSocketPacketCapture : public PacketInput {
public:
  KernelSocketPacketCapture(int port, int buffer_size,
                            int recv_buffer_size = 64 * 1024 * 1024);
  ~KernelSocketPacketCapture();

  void get_packets(ProcessorStateBase &state) override {

    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    // adds a timeout here - otherwise the socket will block indefinitely
    // and get in the way of shutdown.
    struct timeval tv;
    tv.tv_sec = 1; // 1 second timeout
    tv.tv_usec = 0;

    // Make kernel receive buffer a bit larger to avoid dropping packets
    // during the timeout
    setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &recv_buffer_size,
               sizeof(recv_buffer_size));
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    LOG_INFO("Receiver thread started");
    void *next_write_pointer = state.get_current_write_pointer();
    while (state.running) {
      int received = recvfrom(sockfd, next_write_pointer, buffer_size, 0,
                              (struct sockaddr *)&client_addr, &client_len);
      if (received < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
          continue;
        // perror("recvfrom");
        break;
      }

      state.add_received_packet_metadata(received, client_addr);
      state.packets_received += 1;
      next_write_pointer = state.get_next_write_pointer();
    }
    LOG_INFO("Receiver thread exiting");
  };

private:
  int sockfd;
  struct sockaddr_in server_addr;
  int port;
  int buffer_size;
  int recv_buffer_size;
};

class PCAPPacketCapture : public PacketInput {
public:
  PCAPPacketCapture(const std::string &pcap_filename, bool loop = false,
                    uint64_t seq_jump_per_packet = 64);
  ~PCAPPacketCapture();

  void get_packets(ProcessorStateBase &state) override {
    LOG_INFO("PCAP reader thread started for file: {}", filename_);

    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t *handle = nullptr;
    struct pcap_pkthdr *header;
    const u_char *data;
    int res;

    // Statistics
    unsigned long long total_packets = 0;
    unsigned long long total_bytes = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_stats_time = start_time;

    // For sample number incrementing across loops
    uint64_t current_loop = 0;
    uint64_t sample_offset = 0;
    uint64_t min_sample_in_loop = UINT64_MAX;
    uint64_t max_sample_in_loop = 0;
    bool first_loop = true;

    if (loop_ && seq_jump_per_packet_ > 0) {
      LOG_INFO("Sample incrementing enabled with seq jump per packet: {}",
               seq_jump_per_packet_);
    }

    do {
      handle = pcap_open_offline(filename_.c_str(), errbuf);
      if (!handle) {
        LOG_ERROR("Failed to open PCAP file '{}': {}", filename_, errbuf);
        state.running = 0;
        return;
      }

      min_sample_in_loop = UINT64_MAX;
      max_sample_in_loop = 0;

      if (current_loop > 0) {
        LOG_INFO("Starting loop {} with sample offset {}", current_loop,
                 sample_offset);
      } else {
        LOG_INFO("Reading packets from PCAP file...");
      }

      while ((res = pcap_next_ex(handle, &header, &data)) >= 0 &&
             state.running) {
        // Get pointer to write location
        void *write_pointer = state.get_current_write_pointer();

        if (header->caplen <= header->len) {
          std::memcpy(write_pointer, data, header->caplen);

          // Extract and potentially modify the sample number
          // Structure: Ethernet (14) + IP (20) + UDP (8) + CustomHeader
          // CustomHeader layout: sample_count (8), fpga_id (4), freq_channel
          // (2), padding (8)
          const size_t CUSTOM_HEADER_OFFSET = 42; // Ethernet + IP + UDP

          if (header->caplen > CUSTOM_HEADER_OFFSET + sizeof(uint64_t)) {
            // sample_count is the first field in CustomHeader (uint64_t)
            uint64_t *sample_count_ptr = reinterpret_cast<uint64_t *>(
                static_cast<uint8_t *>(write_pointer) + CUSTOM_HEADER_OFFSET);

            uint64_t original_sample = *sample_count_ptr;

            // Track min/max for first loop to calculate range
            if (first_loop) {
              min_sample_in_loop =
                  std::min(min_sample_in_loop, original_sample);
              max_sample_in_loop =
                  std::max(max_sample_in_loop, original_sample);
            }

            // Apply offset for loops after the first
            if (!first_loop && sample_offset > 0) {
              *sample_count_ptr = original_sample + sample_offset;
              LOG_DEBUG("Modified sample count: {} -> {}", original_sample,
                        *sample_count_ptr);
            }
          }

          // Create a fake sender address since PCAP doesn't provide this
          // This might not be necessary.
          struct sockaddr_in client_addr;
          std::memset(&client_addr, 0, sizeof(client_addr));
          client_addr.sin_family = AF_INET;
          client_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
          client_addr.sin_port = htons(0);

          // Add metadata
          state.add_received_packet_metadata(header->caplen, client_addr);
          state.packets_received += 1;

          total_packets++;
          total_bytes += header->caplen;

          // Get next write pointer for next iteration
          while (!state.get_next_write_pointer()) {
            LOG_INFO("[PCAPPacketCapture] Waiting for buffer to become "
                     "available...");
            std::this_thread::sleep_for(std::chrono::microseconds(1));
          }

          // Periodic statistics (every 5 seconds)
          auto now = std::chrono::steady_clock::now();
          auto stats_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                   now - last_stats_time)
                                   .count();

          if (stats_elapsed >= 5) {
            auto total_elapsed =
                std::chrono::duration_cast<std::chrono::seconds>(now -
                                                                 start_time)
                    .count();
            double avg_pps = total_packets / (double)total_elapsed;
            double avg_mbps = (total_bytes * 8.0) / (total_elapsed * 1e6);
            LOG_INFO("PCAP stats: {:.2f} kpps, {:.2f} Mbps (total: {} packets, "
                     "loop: {})",
                     avg_pps / 1000.0, avg_mbps, total_packets, current_loop);
            last_stats_time = now;
          }
        } else {
          LOG_WARN("Packet truncated: caplen={} < len={}", header->caplen,
                   header->len);
        }
      }

      if (res == -1) {
        LOG_ERROR("Error reading PCAP: {}", pcap_geterr(handle));
      } else {
        LOG_INFO("Reached end of PCAP file. Total packets read: {}",
                 total_packets);
      }

      pcap_close(handle);
      handle = nullptr;

      if (loop_ && state.running) {
        // Calculate offset for next loop
        if (first_loop) {
          if (max_sample_in_loop > min_sample_in_loop) {
            // Range of samples in this loop
            uint64_t sample_range = max_sample_in_loop - min_sample_in_loop;
            // Next loop starts at: max + seq_jump_per_packet
            sample_offset = sample_range + seq_jump_per_packet_;

            LOG_INFO("First loop complete. Sample range: {} to {} (span: {})",
                     min_sample_in_loop, max_sample_in_loop, sample_range);
            LOG_INFO("Next loop offset will be: {}", sample_offset);
          } else {
            LOG_WARN("Could not determine sample range, using "
                     "seq_jump_per_packet only");
            sample_offset = seq_jump_per_packet_;
          }
          first_loop = false;
        } else {
          // For subsequent loops, keep adding the same offset
          uint64_t sample_range = max_sample_in_loop - min_sample_in_loop;
          sample_offset += sample_range + seq_jump_per_packet_;
        }

        current_loop++;
        LOG_INFO(
            "Looping back to beginning (loop {}, cumulative offset: {})...",
            current_loop, sample_offset);

        // Small delay before restarting to avoid tight loop
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

    } while (loop_ && state.running);

    LOG_INFO("PCAP reader thread exiting. Total packets: {}, Total bytes: {}, "
             "Loops: {}",
             total_packets, total_bytes, current_loop);
  }

private:
  std::string filename_;
  bool loop_; // Whether to loop the PCAP file
  uint64_t
      seq_jump_per_packet_; // Sequence number jump between packets (e.g., 64)
};
