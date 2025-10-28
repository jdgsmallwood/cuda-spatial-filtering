#pragma once
#include "spatial/logging.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/pipeline_base.hpp"
#include <complex>
#include <cuda.h>
#include <highfive/highfive.hpp>
#include <iostream>
#include <libtcc/Correlator.h>
#include <netinet/in.h>
// #include <sys/socket.h>
#include <atomic>
#include <mutex>

#include <sys/time.h>

#define MIN_PCAP_HEADER_SIZE 64
#define RING_BUFFER_SIZE 1000
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
template <typename T, size_t NR_INPUT_BUFFERS = 2>
class ProcessorState : public ProcessorStateBase {
  // Public member variables
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
    int next_write_index =
        (write_index.load(std::memory_order_relaxed) + 1) % RING_BUFFER_SIZE;
    if (next_write_index == read_index.load(std::memory_order_acquire)) {
      LOG_INFO("Ring buffer is full!! Dropping packets...");
      return false;
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
        break;
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

        break;
      }
    }
  };
  void initialize_buffers(const int first_count) {

    for (auto i = 0; i < NR_INPUT_BUFFERS; ++i) {
      buffers[i].start_seq =
          first_count + i * NR_PACKETS_FOR_CORRELATION * NR_BETWEEN_SAMPLES;
      buffers[i].end_seq =
          first_count +
          ((i + 1) * NR_PACKETS_FOR_CORRELATION - 1) * NR_BETWEEN_SAMPLES;
      buffers[i].is_ready = true;
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
    }
  };

  void execute_processing_pipeline_on_buffer(const int buffer_index) {};

  void release_buffer(const int buffer_index) {
    // I'm imagining this will be called once data has been transferred
    // to the GPU. I don't know exactly how but we'll figure that out I guess.
    // Update old buffer for future use
    //
    // We may well need a mutex or lock here so that multiple GPU threads do not
    // compete.
    std::lock_guard<std::mutex> lock(buffer_index_mutex);
    int max_end_seq_in_buffers = 0;
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
  };

  void advance_to_next_buffer(const int current_buffer_start_seq) {

    // Move to next buffer
    // This is not necessarily the next buffer as the buffers can be
    // updated in any order. This will be the buffer that has the next
    // highest start_seq.
    //  I can probably use a data structure here to pop off the top
    //  rather than checking through everything.
    LOG_INFO("advancing to next buffer...");
    int next_highest_start_seq = -1;
    int next_highest_buffer = -1;
    LOG_INFO("Current buffer start seq is {}", current_buffer_start_seq);
    while (next_highest_start_seq < 0) {
      for (auto i = 0; i < NR_INPUT_BUFFERS; ++i) {
        LOG_INFO("buffer {} start seq is {}.", i, buffers[i].start_seq);
        if (buffers[i].start_seq <= current_buffer_start_seq) {
          LOG_INFO("Continuing...");
          continue;
        };

        if (buffers[i].start_seq < next_highest_start_seq ||
            next_highest_start_seq < 0) {
          LOG_INFO("Buffer {} has higher start_seq {}", i,
                   buffers[i].start_seq);
          next_highest_start_seq = buffers[i].start_seq;
          next_highest_buffer = i;
        };
      }
    };
    current_buffer = next_highest_buffer;

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
    }
  };

  void write_buffer_to_hdf5(const int buffer_index,
                            const std::string &filename) {

    using namespace HighFive;

    // Create / overwrite file
    File file(filename, File::Overwrite);

    // Pointer to your samples
    typename T::PacketSamplesType *samples = d_samples[buffer_index]->samples;

    // Dataset shape
    std::vector<size_t> dims = {T::NR_CHANNELS, NR_PACKETS_FOR_CORRELATION,
                                T::NR_TIME_STEPS_PER_PACKET, T::NR_RECEIVERS,
                                T::NR_POLARIZATIONS};

    // Create dataset of complex<int8_t> (or whatever Sample is)
    DataSet dataset = file.createDataSet<typename T::Sample>("packet_samples",
                                                             DataSpace(dims));

    // Write buffer
    dataset.write(*samples);
  };
  void process_packets() {

    LOG_INFO("Processor thread started");
    //    static bool first_written = false;
    int packets_processed_before_completion_check = 0;
    int current_read_index;
    while (running) {
      while (true) {
        current_read_index = read_index.load(std::memory_order_relaxed);

        if (current_read_index != write_index.load(std::memory_order_acquire)) {
          break;
        }

        if (!running) {
          // Will want to update this to finish buffer that's been processed.
          return;
        }
      }
      typename T::PacketEntryType *entry = d_packet_data[current_read_index];

      if (entry->length == 0 || entry->processed == true) {
        continue;
      }

      process_packet_data(entry);
      read_index.store((current_read_index + 1) % RING_BUFFER_SIZE,
                       std::memory_order_release);
      packets_processed_before_completion_check += 1;
      if (packets_processed_before_completion_check > 100) {
        LOG_INFO("Checking buffer completion...");
        check_buffer_completion();
        if (std::all_of(buffers[current_buffer].is_populated.begin(),
                        buffers[current_buffer].is_populated.end(),
                        [](bool i) { return i; })) {
          // Send off data to be processed by CUDA pipeline.
          // Then advance to next buffer and keep iterating.
          // if (!first_written) {
          //  write_buffer_to_hdf5(current_buffer, "first_buffer.hdf5");
          //  first_written = true;
          //}
          if (pipeline_ == nullptr) {
            throw std::logic_error(
                "Pipeline has not been set. Ensure that set_pipeline has been "
                "called on ProcessorState class.");
          }

          buffers[current_buffer].is_ready = false;
          d_samples[current_buffer]->zero_missing_packets();
          // order here is important. As the pipeline can async update
          // the start/end seqs we need to capture the
          // start_seq, then execute the pipeline, then advance using
          // the saved copy of the start_seq.
          const int current_buffer_start_seq =
              buffers[current_buffer].start_seq;
          pipeline_->execute_pipeline(d_samples[current_buffer]);
          advance_to_next_buffer(current_buffer_start_seq);
        }
        packets_processed_before_completion_check = 0;
      }
    }

    LOG_INFO("Processor thread exiting");
  };

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
      free(d_packet_data[i]);
      d_packet_data[i] = nullptr;
    }

    for (auto i = 0; i < NR_INPUT_BUFFERS; ++i) {
      free(d_samples[i]);
      d_samples[i] = nullptr;
    }
  };
};

class PacketInput {
public:
  virtual void get_packets(ProcessorStateBase &state) = 0;

  virtual ~PacketInput() = default;
};

class KernelSocketPacketCapture : public PacketInput {
public:
  KernelSocketPacketCapture(int port, int buffer_size);
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
    int recv_buffer_size = 8 * 1024 * 1024;
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
        perror("recvfrom");
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
};
