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
#include <barrier>
#include <bitset>
#include <chrono>
#include <mutex>
#include <pcap/pcap.h>
#include <queue>
#include <stdatomic.h>
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
template <size_t NR_CHANNELS, size_t NR_FPGA_SOURCES> struct BufferState {
  bool is_ready;
  std::array<unsigned long long, NR_FPGA_SOURCES> start_seq;
  std::array<unsigned long long, NR_FPGA_SOURCES> end_seq;
  std::bitset<NR_CHANNELS> is_populated;
};

// forward declaration of GPUPipeline.
class GPUPipeline;

class ProcessorStateBase {
public:
  int current_buffer = 0;
  std::atomic<int> write_index = 0;
  std::atomic<int> read_index = 0;
  std::unordered_map<uint32_t, int> fpga_ids;
  bool synchronous_pipeline = false;
  std::atomic<int> running = 1;
  unsigned long long packets_received = 0;
  std::atomic<unsigned long long> packets_processed = 0;
  unsigned long long packets_missing = 0;
  std::atomic<unsigned long long> packets_discarded = 0;
  unsigned long long pipeline_runs_queued = 0;
  virtual void *get_next_write_pointer() = 0;
  virtual void *get_current_write_pointer() = 0;
  virtual void add_received_packet_metadata(const int length,
                                            const sockaddr_in &client_addr) = 0;
  virtual void release_buffer(const int buffer_index) = 0;
  virtual void set_pipeline(GPUPipeline *pipeline) = 0;
  virtual void process_all_available_packets() = 0;

  virtual void handle_buffer_completion(bool force_flush = false) = 0;
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

  std::array<BufferState<T::NR_CHANNELS, T::NR_FPGA_SOURCES>, NR_INPUT_BUFFERS>
      buffers;
  uint64_t latest_packet_received[T::NR_CHANNELS][T::NR_FPGA_SOURCES] = {};
  mutable std::mutex buffer_index_mutex;
  GPUPipeline *pipeline_;
  // Constructor / Destructor
  ProcessorState(
      size_t nr_packets_for_correlation, size_t nr_between_samples,
      size_t min_freq_channel,
      std::unique_ptr<std::unordered_map<uint32_t, int>> fpga_ids_ = nullptr)
      : NR_PACKETS_FOR_CORRELATION(nr_packets_for_correlation),
        NR_BETWEEN_SAMPLES(nr_between_samples),
        MIN_FREQ_CHANNEL(min_freq_channel) {

    if (fpga_ids_ && !fpga_ids_->empty()) {
      fpga_ids = *fpga_ids_;
    } else {
      for (int i = 0; i < T::NR_FPGA_SOURCES; ++i) {
        fpga_ids[i] = i;
      }
    }
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
    LOG_DEBUG("Next write index is...{}", next_write_index);
    return true;
  };

  __attribute__((hot)) void copy_data_to_input_buffer_if_able(
      ProcessedPacket<typename T::PacketScaleStructure,
                      typename T::PacketDataStructure> &pkt,
      const int current_read_index,
      const std::array<unsigned long long, T::NR_FPGA_SOURCES> &global_max) {
    size_t fpga_index;
    if (fpga_ids.count(pkt.fpga_id)) {
      fpga_index = fpga_ids[pkt.fpga_id];
    } else {
      LOG_ERROR("FPGA ID {} not found.", pkt.fpga_id);
      throw std::out_of_range("FPGA ID not found. Check logs.");
    }

    const int freq_channel = pkt.freq_channel - MIN_FREQ_CHANNEL;

    {
      std::lock_guard lock(latest_packet_mutex);
      latest_packet_received[freq_channel][fpga_index] = std::max(
          latest_packet_received[freq_channel][fpga_index], pkt.sample_count);
    }
    const int current_buf = current_buffer;
    const uint64_t sample_count = pkt.sample_count;
    // on the first run global_max will not be set initially so will be 0.
    // We don't want it to seize up on this.
    if (sample_count > global_max[fpga_index] && global_max[fpga_index] > 0) {
      std::lock_guard lock(future_packet_queue_mutex);
      future_packet_queue[fpga_index].push({current_read_index, sample_count});
      return;
    }
    // copy to correct place or leave it.
    for (int buffer = 0; buffer < NR_INPUT_BUFFERS; ++buffer) {
      const int buffer_index = (current_buf + buffer) % NR_INPUT_BUFFERS;
      const unsigned long long buffer_start =
          buffers[buffer_index].start_seq[fpga_index];
      const int packet_index =
          (sample_count - buffer_start) / NR_BETWEEN_SAMPLES;

      if (buffer == 0 && packet_index < 0) [[unlikely]] {
        // This means that this packet is less than the lowest possible
        // start token. Maybe an out-of-order packet that's coming in?
        // Regardless we can't do anything with this.
        LOG_INFO("Discarding packet as it is before current buffer with "
                 "begin_seq {} actually has packet_index {}",
                 buffer_start, pkt.sample_count);
        packets_discarded.fetch_add(1);
        *pkt.original_packet_processed = true;
        return;
      }

      if (packet_index >= 0 && packet_index < NR_PACKETS_FOR_CORRELATION) {
        const int receiver_index = fpga_index * T::NR_RECEIVERS_PER_PACKET;
        // LOG_DEBUG("Copying data to packet_index {} and channel index {} and "
        //           "receiver_index {} of buffer {}",
        //           packet_index, freq_channel, receiver_index, buffer_index);

        auto &buffer = d_samples[buffer_index];
        auto &samples =
            (*buffer->samples)[freq_channel][packet_index][fpga_index];
        auto &scales =
            (*buffer->scales)[freq_channel][packet_index][receiver_index];
        auto &arrival =
            buffer->arrivals[0][freq_channel][packet_index][fpga_index];
        std::memcpy(&samples, pkt.payload->data,
                    sizeof(typename T::PacketDataStructure));
        std::memcpy(&scales, pkt.payload->scales,
                    sizeof(typename T::PacketScaleStructure));
        arrival = true;
        // LOG_DEBUG("Setting original_packet_processed as true...");
        // LOG_DEBUG("original_packet_processed_before={}",
        //           *pkt.original_packet_processed);
        *(pkt.original_packet_processed) = true;
        // LOG_DEBUG("DEBUG: original_packet_processed_after={}",
        //           *pkt.original_packet_processed);

        return;
      }
    }
    // LOG_DEBUG(
    //     "Packet with seq number {} and read index {} was unable to find a "
    //     "home. Adding to "
    //     "future_packet_queue...",
    //     sample_count, current_read_index);
  };
  void process_all_available_packets() {
    // This exists mainly for testing purposes
    // to allow us to add some packets then process them all.
    const std::array<unsigned long long, T::NR_FPGA_SOURCES> global_max;
    for (int i = 0; i < T::NR_FPGA_SOURCES; ++i) {
      global_max[i] = global_max_end_seq[i].load(std::memory_order_acquire);
    }
    while (read_index.load() != write_index.load()) {
      int current_read_index = read_index.load();
      process_packet_data(d_packet_data[current_read_index], current_read_index,
                          global_max);
      int new_read_index = (current_read_index + 1) % RING_BUFFER_SIZE;
      read_index.store(new_read_index, std::memory_order_release);
    }
  }

  void initialize_buffers(const unsigned long long first_count,
                          const uint32_t fpga_id) {
    LOG_INFO("[BufferInitialization] First count for FPGA ID {} was {}...",
             fpga_id, first_count);
    std::lock_guard lock(buffer_index_mutex);
    const int fpga_index = fpga_ids[fpga_id];
    for (auto i = 0; i < NR_INPUT_BUFFERS; ++i) {
      buffers[i].start_seq[fpga_index] =
          first_count + i * NR_PACKETS_FOR_CORRELATION * NR_BETWEEN_SAMPLES;
      buffers[i].end_seq[fpga_index] =
          first_count +
          ((i + 1) * NR_PACKETS_FOR_CORRELATION - 1) * NR_BETWEEN_SAMPLES;
      buffers[i].is_ready = true;
      LOG_INFO("[BufferInitialization] Buffer {} goes from {} to {}", i,
               buffers[i].start_seq[fpga_index],
               buffers[i].end_seq[fpga_index]);
      // we know 0 will be the first one - so no need to add zero
      if (i != 0 && fpga_index == 0) {
        // can just use the first FPGA for checking which buffer should
        // come next.
        buffer_ordering_queue.push({i, buffers[i].start_seq[0]});
      };
    }
    global_max_end_seq[fpga_index].store(
        buffers[NR_INPUT_BUFFERS - 1].end_seq[fpga_index],
        std::memory_order_release);
  };

  __attribute__((hot)) void process_packet_data(
      typename T::PacketEntryType *pkt, const int current_read_index,
      const std::array<unsigned long long, T::NR_FPGA_SOURCES> &global_max) {

    // This is where you'd do your actual processing
    // For now, just print the info and simulate some work

    ProcessedPacket parsed = pkt->parse();

    if (pkt->processed) [[unlikely]] {
      return;
    }
    // LOG_DEBUG(
    //     "Processing packet: sample_count={}, freq_channel={}, fpga_id={}, "
    //     "payload={} bytes",
    //     parsed.sample_count, parsed.freq_channel, parsed.fpga_id,
    //     parsed.payload_size);

    // LOG_DEBUG("First data point...{} + {} i",
    //           parsed.payload->data[0][0][0].real(),
    //           parsed.payload->data[0][0][0].imag());

    std::call_once(buffer_init_flag[fpga_ids[pkt.fpga_id]], [&]() {
      LOG_INFO("Initializing buffers as this is the first packet...");
      initialize_buffers(parsed.sample_count, pkt.fpga_id);
    });
    copy_data_to_input_buffer_if_able(parsed, current_read_index, global_max);
    if (*parsed.original_packet_processed) [[likely]] {
      packets_processed.fetch_add(1);
    }
    modified_since_last_completion_check[parsed.freq_channel -
                                         MIN_FREQ_CHANNEL] = true;
  };

  void get_global_max_packet_array(
      std::array<unsigned long long, T::NR_FPGA_SOURCES> &arr) {
    for (int i = 0; i < T::NR_FPGA_SOURCES; ++i) {
      arr[i] = global_max_end_seq[i].load(std::memory_order_acquire);
    }
  }

  void execute_processing_pipeline_on_buffer(const int buffer_index) {};

  __attribute__((hot)) void release_buffer(const int buffer_index) {
    // LOG_INFO("[ProcessorState] Releasing buffer with index {}",
    // buffer_index);
    //  This is called to let the processor know that the buffer has been
    //  copied to the GPU and now can be overwritten.
    // This is necessary to avoid multiple GPU threads competing / racing.
    // LOG_DEBUG(
    //    "[ProcessorState - release_buffer] acquiring lock for index {}...",
    //    buffer_index);
    {
      std::lock_guard<std::mutex> lock(buffer_index_mutex);

      std::array<unsigned long long, T::NR_FPGA_SOURCES> max_end_seq_in_buffers;
      get_global_max_packet_array(max_end_seq_in_buffers);
      const int buf_idx = buffer_index;
      auto &buffer = buffers[buf_idx];
      for (int i = 0; i < T::NR_FPGA_SOURCES; ++i) {
        const unsigned long long new_start =
            max_end_seq_in_buffers[i] + 1 * NR_BETWEEN_SAMPLES;
        const unsigned long long new_end =
            new_start + (NR_PACKETS_FOR_CORRELATION - 1) * NR_BETWEEN_SAMPLES;
        // LOG_DEBUG("[ProcessorState - release_buffer] lock acquired for index
        // {}...",
        //           buffer_index);
        // for (auto i = 0; i < NR_INPUT_BUFFERS; ++i) {
        //  max_end_seq_in_buffers =
        //      std::max(max_end_seq_in_buffers, buffers[i].end_seq);
        //}
        buffer.start_seq[i] = new_start;
        buffer.end_seq[i] = new_end;
        global_max_end_seq[i].store(new_end, std::memory_order_release);
      }
      // LOG_DEBUG("[ProcessorState - release_buffer] filling arrivals as false
      // for "
      //           "buffer {}...",
      //           buffer_index);
      std::memset(d_samples[buf_idx]->arrivals, 0,
                  T::NR_CHANNELS * T::NR_PACKETS_FOR_CORRELATION *
                      T::NR_FPGA_SOURCES * sizeof(bool));
      buffer.is_populated.reset();
      // LOG_DEBUG("[ProcessorState - release_buffer] pushing to queue for index
      // {} "
      //           "with seq {}",
      //           buffer_index, buffers[buffer_index].start_seq);

      buffer.is_ready = true;
      buffer_ordering_queue.push({buf_idx, buffer.start_seq[0]});
    }
    buffer_available_cv.notify_one();
    // LOG_INFO("[ProcessorState] added buffer index {} back to queue with "
    //          "start_seq {}",
    //          buffer_index, buffers[buffer_index].start_seq);
  };
  __attribute__((hot)) int advance_to_next_buffer() {

    // Move to next buffer
    // This is not necessarily the next buffer as the buffers can be
    // updated in any order. This will be the buffer that has the next
    // highest start_seq.
    // LOG_INFO("advancing to next buffer...");
    BufferOrder b;
    std::unique_lock<std::mutex> lock(buffer_index_mutex);
    buffer_available_cv.wait(lock,
                             [this] { return !buffer_ordering_queue.empty(); });
    b = buffer_ordering_queue.top();
    buffer_ordering_queue.pop();
    lock.unlock();
    // LOG_INFO("Current buffer start seq is {}", current_buffer_start_seq);
    current_buffer = b.index;

    // LOG_INFO(
    //     "next current_buffer is {} and has start_seq {} and is it ready? {}",
    //     current_buffer, buffers[current_buffer].start_seq,
    //     buffers[current_buffer].is_ready);

    // this is kinda assuming there's an async callback that will
    // ready the buffer after the data has been transferred out.
    // This async callback will come from the GPUPipeline class.
    // Reset new current buffer
    // This is just NR_CHANNELS_DEF booleans that tell us if the current
    // buffer has all the data it needs.
    buffers[b.index].is_populated.reset();
    // Set modified to true so that all channels get checked on next
    // completion check in case there are packets that went ahead.
    std::fill(modified_since_last_completion_check.begin(),
              modified_since_last_completion_check.end(), true);
    //  LOG_INFO(
    //      "Current buffer is all complete. Moving to next buffer which is
    //      #{}", current_buffer);
    //  LOG_INFO("New buffer starts at packet {} and ends at {}",
    //           buffers[current_buffer].start_seq,
    //           buffers[current_buffer].end_seq);
    return b.index;
  }

  void check_buffer_completion(std::vector<int> &buffers_complete) {

    const int current_buf = current_buffer;
    for (int i = 0; i < NR_INPUT_BUFFERS; ++i) {
      // i want current buffer to be the first one in the list to preserve
      // ordering.
      const int buf_idx = (current_buf + i) % NR_INPUT_BUFFERS;
      auto &buffer = buffers[buf_idx];
      if (!buffer.is_ready) {
        continue;
      }
      const std::array<unsigned long long, T::NR_FPGA_SOURCES> end_seq =
          buffer.end_seq;
      for (auto channel = 0; channel < T::NR_CHANNELS; ++channel) {
        if (buffer.is_populated[channel] ||
            !modified_since_last_completion_check[channel]) {
          continue;
        }
        // LOG_INFO("Check if buffers are complete for channel {}", channel);
        bool all_fpgas_complete = true;
        for (int fpga = 0; fpga < T::NR_FPGA_SOURCES; ++fpga) {
          // we wait for halfway through the next buffer to be complete to avoid
          // missing out of order packets.
          if (latest_packet_received[channel][fpga] <
              end_seq[fpga] + NR_BETWEEN_SAMPLES / 2) {
            all_fpgas_complete = false;
            break;
          }
        }
        if (all_fpgas_complete) {
          buffer.is_populated[channel] = true;
        }
        // LOG_INFO("Buffer is complete for channel {}", channel);
        // else {
        //  LOG_INFO("Buffer is not complete for channel {} as end_seq is {} and
        //  "
        //          "latest_packet_receives are:",
        //         channel, buffers[current_buffer].end_seq);
        // for (int check = 0; check < T::NR_FPGA_SOURCES; ++check) {
        //   LOG_INFO("FPGA ID {} / Channel {}: {},", check, channel,
        //            latest_packet_received[channel][check]);
        // }
        // }
      }
      if (buffer.is_populated.all()) {
        buffers_complete.push_back(buf_idx);
      }
    }
    for (int channel = 0; channel < T::NR_CHANNELS; channel++) {
      modified_since_last_completion_check[channel] = false;
    }
  };

  __attribute__((hot)) void process_packets() {

    // using clock = std::chrono::high_resolution_clock;
    // auto cpu_start = clock::now();
    // auto cpu_end = clock::now();
    LOG_INFO("Processor thread started");
    start_processing_threads();
    int current_read_index;
    constexpr int num_loops_before_completion_check = 1;
    int packets_until_completion_check = num_loops_before_completion_check;
    current_read_index = read_index.load(std::memory_order_relaxed);

    constexpr int REGULAR_BATCH_SIZE = 600;
    while (running.load(std::memory_order_acquire)) [[likely]] {
      const int current_write_index =
          write_index.load(std::memory_order_acquire);

      int slice_end = current_read_index;
      for (auto i = 0; i < REGULAR_BATCH_SIZE; ++i) {
        if (slice_end == current_write_index)
          break;
        slice_end = (slice_end + 1) % RING_BUFFER_SIZE;
      }

      int slice_len = (slice_end - current_read_index + RING_BUFFER_SIZE) %
                      RING_BUFFER_SIZE;
      int per_worker = (slice_len + WORKER_COUNT - 1) / WORKER_COUNT;
      int start = current_read_index;

      {
        std::lock_guard<std::mutex> lock(work_mutex);
        int workers_with_tasks = 0;
        for (auto i = 0; i < WORKER_COUNT; ++i) {
          int worker_start = start;
          int worker_end = start;

          for (int j = 0; j < per_worker; ++j) {
            if (worker_end == slice_end)
              break;
            worker_end = (worker_end + 1) % RING_BUFFER_SIZE;
          }

          if (worker_start == worker_end) {
            worker_has_task[i].store(false, std::memory_order_release);
          } else {
            worker_tasks[i] = {worker_start, worker_end};
            worker_has_task[i].store(true, std::memory_order_release);
            worker_completed_task[i].store(false, std::memory_order_release);
            workers_with_tasks++;
          }
          start = worker_end;
        }
        num_workers_with_tasks.store(workers_with_tasks,
                                     std::memory_order_release);
      }
      work_cv.notify_all();
      {
        std::unique_lock<std::mutex> lock(work_mutex);
        work_cv.wait(lock, [this] {
          return !running.load() || num_workers_with_tasks.load() == 0;
        });
      }

      current_read_index = slice_end;

      read_index.store(current_read_index, std::memory_order_release);

      if (--packets_until_completion_check == 0) {
        // Before checking buffer completion drain any packets from the
        // queue that should be in this buffer
        const std::array<unsigned long long, T::NR_FPGA_SOURCES> global_max;
        get_global_max_packet_array(global_max);
        {
          std::unique_lock<std::mutex> lock(future_packet_queue_mutex,
                                            std::try_to_lock);
          if (lock.owns_lock()) {
            for (int i = 0; i < T::NR_FPGA_SOURCES; ++i) {
              while (!future_packet_queue[i].empty()) {
                auto pkt = future_packet_queue[i].top();
                if (pkt.packet_num > global_max[i]) {
                  break;
                }
                future_packet_queue[i].pop();
                lock.unlock();
                auto *entry = d_packet_data[pkt.index];
                if (entry->length > 0 && !entry->processed) {
                  process_packet_data(entry, pkt.index, global_max);
                }
                lock.lock();
              }
            }
          }
        }
        //    cpu_start = clock::now();
        handle_buffer_completion();
        //  cpu_end = clock::now();
        packets_until_completion_check = num_loops_before_completion_check;
        // LOG_DEBUG("CPU time for buffer completion check: {} us",
        //           std::chrono::duration_cast<std::chrono::microseconds>(
        //               cpu_end - cpu_start)
        //               .count());
      }
    }
    // shut down pipeline thread
    buffer_ready_for_pipeline.notify_all();
    stop_processing_threads();
    LOG_INFO("Processor thread exiting");
  };

  __attribute__((hot)) void worker_thread_fn(int worker_id) {
    std::cout << "Starting worker with id " << worker_id << std::endl;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(worker_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    while (running.load(std::memory_order_acquire)) {
      std::unique_lock<std::mutex> lock(work_mutex);
      work_cv.wait(lock, [&] {
        return !running.load() || worker_has_task[worker_id].load();
      });
      if (!running.load(std::memory_order_acquire)) {
        std::cout << "Worker " << worker_id << " exiting at the top\n";
        break;
      }

      const std::array<unsigned long long, T::NR_FPGA_SOURCES> global_max;
      get_global_max_packet_array(global_max);
      auto [start, end] = worker_tasks[worker_id];
      lock.unlock();
      // std::cout << "Worker with id " << worker_id
      //           << " task started with start " << start << " and end " <<
      //           end
      //           << std::endl;
      int idx = start;

      while (idx != end) {

        auto *entry = d_packet_data[idx];

        const int next_idx = (idx + 1) % RING_BUFFER_SIZE;
        __builtin_prefetch(d_packet_data[next_idx], 0, 3);
        __builtin_prefetch(d_packet_data[next_idx]->data, 0, 3);
        __builtin_prefetch(d_packet_data[next_idx]->data + 12, 0, 3);
        __builtin_prefetch(d_packet_data[next_idx]->data + 42, 0, 3);

        if (entry->length > 0 && !entry->processed) {
          process_packet_data(entry, idx, global_max);
        }
        idx = (idx + 1) % RING_BUFFER_SIZE;
      }

      lock.lock();
      worker_completed_task[worker_id].store(true);
      worker_has_task[worker_id].store(false);
      num_workers_with_tasks.fetch_sub(1);
      work_cv.notify_all();
    }
  }

  __attribute__((hot)) void handle_buffer_completion(bool force_flush = false) {

    if (pipeline_ == nullptr) [[unlikely]] {
      throw std::logic_error(
          "Pipeline has not been set. Ensure that set_pipeline has been "
          "called on ProcessorState class.");
    }
    using clock = std::chrono::high_resolution_clock;
    auto cpu_start = clock::now();
    auto cpu_end = clock::now();
    // LOG_INFO("Checking buffer completion...");
    std::vector<int> buffers_complete;
    check_buffer_completion(buffers_complete);

    if (force_flush && buffers_complete.empty()) {
      buffers_complete.push_back(current_buffer);
    }

    if (buffers_complete.empty()) {
      return;
    }

    int current_buf = buffers_complete[0];
    for (int i = 0; i < buffers_complete.size(); ++i) {
      LOG_INFO("pipeline feeding {}", current_buf);
      auto &buffer = buffers[current_buf];

      if (std::find(buffers_complete.begin(), buffers_complete.end(),
                    current_buf) == buffers_complete.end()) [[unlikely]] {
        return;
      }
      // LOG_INFO("Buffer is complete - passing to output pipeline...");
      //  Send off data to be processed by CUDA pipeline.
      //  Then advance to next buffer and keep iterating.

      buffer.is_ready = false;
      cpu_start = clock::now();
      // LOG_INFO("Zeroing missing packets...");
      d_samples[current_buf]->zero_missing_packets();
      packets_missing += d_samples[current_buf]->get_num_missing_packets();
      cpu_end = clock::now();
      // LOG_DEBUG("CPU time for zeroing packets: {} us",
      //           std::chrono::duration_cast<std::chrono::microseconds>(cpu_end
      //           -
      //                                                                 cpu_start)
      //               .count());
      //  order here is important. As the pipeline can async update
      //  the start/end seqs we need to capture the
      //  start_seq, then execute the pipeline, then advance using
      //  the saved copy of the start_seq.
      // LOG_INFO("Enqueueing buffer {} for pipeline...", current_buffer);
      {
        if (!synchronous_pipeline) [[likely]] {
          {
            std::lock_guard<std::mutex> lock(buffers_ready_for_pipeline_lock);
            buffers_ready_for_pipeline.push(current_buf);
          }
          buffer_ready_for_pipeline.notify_one();
        } else [[unlikely]] {
          pipeline_->execute_pipeline(d_samples[current_buf]);
          pipeline_runs_queued += 1;
        }
      }
      cpu_start = clock::now();
      current_buf = advance_to_next_buffer();
      cpu_end = clock::now();
    }
    // LOG_DEBUG("CPU time for advancing to next buffer: {} us",
    //           std::chrono::duration_cast<std::chrono::microseconds>(cpu_end
    //           -
    //                                                                 cpu_start)
    //               .count());
    // LOG_INFO("Done!");
  };

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(buffers_ready_for_pipeline_lock);
      std::lock_guard<std::mutex> lock_stop(work_mutex);
      running.store(0, std::memory_order_release);
    };
    buffer_ready_for_pipeline.notify_all();
    work_cv.notify_all();
  };

  void pipeline_feeder() {
    LOG_INFO("Pipeline feeder starting up...");
    while (running.load(std::memory_order_acquire) == 1) {
      std::unique_lock<std::mutex> lock(buffers_ready_for_pipeline_lock);
      buffer_ready_for_pipeline.wait(lock, [&] {
        return !buffers_ready_for_pipeline.empty() ||
               (running.load(std::memory_order_acquire) == 0);
      });

      if (running.load(std::memory_order_acquire) == 0) {
        break;
      }
      size_t buffer_index = buffers_ready_for_pipeline.front();
      buffers_ready_for_pipeline.pop();
      lock.unlock();
      LOG_INFO("Buffer index {} picked up by pipeline feeder...", buffer_index);
      pipeline_->execute_pipeline(d_samples[buffer_index]);
      pipeline_runs_queued += 1;
    }
    LOG_INFO("Pipeline feeder exiting!");
  }

  void *get_current_write_pointer() {
    return (void *)&(d_packet_data[write_index]->data);
  }
  void *get_next_write_pointer() {
    while (!get_next_write_index() && running) {
      LOG_DEBUG("Waiting for next pointer....");
      // std::this_thread::sleep_for(std::chrono::nanoseconds(10));
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
    unsigned long long start_seq;

    // Compare to make the priority queue a min-heap based on start_seq
    bool operator>(const BufferOrder &other) const {
      return start_seq > other.start_seq;
    }
  };

  struct PacketOrder {
    int index;
    unsigned long long packet_num;

    bool operator>(const PacketOrder &other) const {
      return packet_num > other.packet_num;
    }
  };

  std::queue<size_t> buffers_ready_for_pipeline;
  mutable std::mutex buffers_ready_for_pipeline_lock;
  std::condition_variable buffer_ready_for_pipeline;
  // This is for packets that arrive but their buffer is not yet created.
  // The read pointer moves on but keep these as things to process.
  // They will not be overwritten by the write pointer as it checks whether or
  // not they have been processed.
  std::array<std::priority_queue<PacketOrder, std::vector<PacketOrder>,
                                 std::greater<PacketOrder>>,
             T::NR_FPGA_SOURCES>
      future_packet_queue;
  std::mutex future_packet_queue_mutex;
  std::array<bool, T::NR_CHANNELS> modified_since_last_completion_check;
  std::priority_queue<BufferOrder, std::vector<BufferOrder>,
                      std::greater<BufferOrder>>
      buffer_ordering_queue;
  std::condition_variable buffer_available_cv;
  std::array<std::atomic<unsigned long long>, T::NR_FPGA_SOURCES>
      global_max_end_seq{0};
  std::array<std::once_flag, T::NR_FPGA_SOURCES> buffer_init_flag;

  std::mutex latest_packet_mutex;
  static constexpr int WORKER_COUNT = 3;
  struct WorkRange {
    int start;
    int end;
  };

  std::array<std::atomic<bool>, WORKER_COUNT> worker_has_task;
  std::array<std::atomic<bool>, WORKER_COUNT> worker_completed_task;
  std::array<WorkRange, WORKER_COUNT> worker_tasks;

  std::mutex work_mutex;
  std::condition_variable work_cv;
  std::atomic<int> num_workers_with_tasks = 0;

  std::vector<std::thread> workers;

  void start_processing_threads() {
    for (int i = 0; i < WORKER_COUNT; i++) {
      worker_has_task[i].store(false);
      workers.emplace_back(&ProcessorState::worker_thread_fn, this, i);
    };
  };
  void stop_processing_threads() {
    work_cv.notify_all();
    for (auto &t : workers) {
      t.join();
    }
  }
};

class PacketInput {
public:
  virtual void get_packets(ProcessorStateBase &state) = 0;

  virtual ~PacketInput() = default;
};

class KernelSocketPacketCapture : public PacketInput {
public:
  KernelSocketPacketCapture(std::string &ifname, int port, int buffer_size,
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
  std::string ifname;
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
          state.get_next_write_pointer();

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
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
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

class PCAPMultiFPGAPacketCapture : public PacketInput {
public:
  PCAPMultiFPGAPacketCapture(const std::string &pcap_filename,
                             bool loop = false, int num_fpgas = 4,
                             uint64_t seq_jump_per_packet = 64);
  ~PCAPMultiFPGAPacketCapture();

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

      if (current_loop > 0) {
        LOG_INFO("Starting loop {} with sample offset {}", current_loop,
                 sample_offset);
      } else {
        LOG_INFO("Reading packets from PCAP file...");
      }

      while ((res = pcap_next_ex(handle, &header, &data)) >= 0 &&
             state.running) {
        // Get pointer to write location
        for (int i = 0; i < num_fpgas; ++i) {
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
              uint8_t *custom_header_start =
                  static_cast<uint8_t *>(write_pointer) + CUSTOM_HEADER_OFFSET;
              uint64_t *sample_count_ptr =
                  reinterpret_cast<uint64_t *>(custom_header_start);

              uint32_t *fpga_id_ptr =
                  reinterpret_cast<uint32_t *>(custom_header_start + 8);

              *fpga_id_ptr = i;
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
            state.get_next_write_pointer();
          } else {
            LOG_WARN("Packet truncated: caplen={} < len={}", header->caplen,
                     header->len);
          }
        }
        // Periodic statistics (every 5 seconds)
        auto now = std::chrono::steady_clock::now();
        auto stats_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                 now - last_stats_time)
                                 .count();

        if (stats_elapsed >= 5) {
          auto total_elapsed =
              std::chrono::duration_cast<std::chrono::seconds>(now - start_time)
                  .count();
          double avg_pps = total_packets / (double)total_elapsed;
          double avg_mbps = (total_bytes * 8.0) / (total_elapsed * 1e6);
          LOG_INFO("PCAP stats: {:.2f} kpps, {:.2f} Mbps (total: {} packets, "
                   "loop: {})",
                   avg_pps / 1000.0, avg_mbps, total_packets, current_loop);
          last_stats_time = now;
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
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
  int num_fpgas;
};
