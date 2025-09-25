#pragma once
#include "spatial/packet_formats.hpp"
#include "spatial/tcc_config.h"
#include <complex>
#include <cuda.h>
#include <highfive/highfive.hpp>
#include <iostream>
#include <libtcc/Correlator.h>
#include <netinet/in.h>
// #include <sys/socket.h>
#include <atomic>

#include <sys/time.h>
#ifndef NR_BEAMS
#define NR_BEAMS 2
#endif

#ifndef NR_PACKETS_FOR_CORRELATION
#define NR_PACKETS_FOR_CORRELATION 16
#endif

#ifndef NR_TIME_STEPS_PER_PACKET
#define NR_TIME_STEPS_PER_PACKET 64
#endif

#ifndef NR_ACTUAL_RECEIVERS
#define NR_ACTUAL_RECEIVERS 20
#endif

#ifndef NR_BUFFERS
#define NR_BUFFERS 2
#endif

#ifndef NR_CORRELATION_BLOCKS_TO_INTEGRATE
#define NR_CORRELATION_BLOCKS_TO_INTEGRATE 10
#endif

#define MIN_PCAP_HEADER_SIZE 64
#define RING_BUFFER_SIZE 1000
#define BUFFER_SIZE 4096
#include <cuda_fp16.h>

namespace spatial {

constexpr int NR_TIMES_PER_BLOCK = 128 / NR_BITS;
constexpr int NR_BASELINES = NR_RECEIVERS * (NR_RECEIVERS + 1) / 2;
constexpr int NR_ACTUAL_BASELINES =
    NR_ACTUAL_RECEIVERS * (NR_ACTUAL_RECEIVERS + 1) / 2;
constexpr int NR_BLOCKS_FOR_CORRELATION =
    NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET / NR_TIMES_PER_BLOCK;
constexpr int NR_TIME_STEPS_FOR_CORRELATION =
    NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET;

} // namespace spatial

#if NR_BITS == 4
typedef complex_int4_t Sample;
typedef std::complex<int32_t> Visibility;
constexpr tcc::Format inputFormat = tcc::Format::i4;
#define CAST_TO_FLOAT(x) (x)
#elif NR_BITS == 8
typedef std::complex<int8_t> Sample;
typedef std::complex<int32_t> Visibility;
typedef int8_t Tin;
typedef int16_t Tscale;
constexpr tcc::Format inputFormat = tcc::Format::i8;
#define CAST_TO_FLOAT(x) (x)
#elif NR_BITS == 16
typedef std::complex<__half> Sample;
typedef std::complex<float> Visibility;
typedef __half Tin;
typedef float Tscale;
#define CAST_TO_FLOAT(x) __half2float(x)
constexpr tcc::Format inputFormat = tcc::Format::fp16;
#endif
typedef std::complex<float> FloatVisibility;
typedef Sample Samples[NR_CHANNELS][spatial::NR_BLOCKS_FOR_CORRELATION]
                      [NR_RECEIVERS][NR_POLARIZATIONS]
                      [spatial::NR_TIMES_PER_BLOCK];
typedef std::complex<__half>
    HalfSamples[NR_CHANNELS][spatial::NR_BLOCKS_FOR_CORRELATION][NR_RECEIVERS]
               [NR_POLARIZATIONS][spatial::NR_TIMES_PER_BLOCK];
typedef Visibility Visibilities[NR_CHANNELS][spatial::NR_BASELINES]
                               [NR_POLARIZATIONS][NR_POLARIZATIONS];
typedef std::complex<float>
    FloatVisibilities[NR_CHANNELS][spatial::NR_BASELINES][NR_POLARIZATIONS]
                     [NR_POLARIZATIONS];
typedef std::complex<__half> BeamWeights[NR_CHANNELS][NR_POLARIZATIONS]
                                        [NR_BEAMS][NR_RECEIVERS];

typedef std::complex<float>
    BeamformedData[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS]
                  [spatial::NR_TIME_STEPS_FOR_CORRELATION];

template <typename T>
void eigendecomposition(float *h_eigenvalues, int n, const std::vector<T> *A);
template <typename T>
void d_eigendecomposition(float *d_eigenvalues, const int n,
                          const int num_channels, const int num_polarizations,
                          T *d_A, cudaStream_t stream);
void correlate(Samples *samples, Visibilities *visibilities);
void ccglib_mma(__half *A, __half *B, float *C, const int n_row,
                const int n_col, const int batch_size, int n_inner = -1);

void ccglib_mma_opt(__half *A, __half *B, float *C, const int n_row,
                    const int n_col, const int batch_size, int n_inner,
                    const int tile_size_x, const int tile_size_y);
void beamform(Samples *h_samples, std::complex<__half> *h_weights,
              BeamformedData *h_beam_output,
              FloatVisibilities *h_visibilities_output,
              const int nr_aggregated_packets);
template <typename T>
void rearrange_matrix_to_ccglib_format(const std::complex<T> *input_matrix,
                                       T *output_matrix, const int n_rows,
                                       const int n_cols,
                                       const bool row_major = true);

template <typename T>
void rearrange_ccglib_matrix_to_compact_format(const T *input_matrix,
                                               std::complex<T> *output_matrix,
                                               const int n_rows,
                                               const int n_cols);

inline void checkCudaCall(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << "error " << error << std::endl;
    exit(1);
  }
}

inline void print_nonzero_visibilities(const Visibilities *vis) {
  for (int ch = 0; ch < NR_CHANNELS; ++ch) {
    for (int bl = 0; bl < spatial::NR_BASELINES; ++bl) {
      for (int pol1 = 0; pol1 < NR_POLARIZATIONS; ++pol1) {
        for (int pol2 = 0; pol2 < NR_POLARIZATIONS; ++pol2) {
          const Visibility v = (*vis)[ch][bl][pol1][pol2];
          if (v.real() != 0.0f || v.imag() != 0.0f) {
            std::cout << "vis[" << ch << "][" << bl << "][" << pol1 << "]["
                      << pol2 << "] = (" << v.real() << ", " << v.imag()
                      << ")\n";
          }
        }
      }
    }
  }
}

inline void print_nonzero_visibilities(const Visibilities *vis,
                                       const Tscale *scales) {
  for (int ch = 0; ch < NR_CHANNELS; ++ch) {
    for (int bl = 0; bl < spatial::NR_BASELINES; ++bl) {
      for (int pol1 = 0; pol1 < NR_POLARIZATIONS; ++pol1) {
        for (int pol2 = 0; pol2 < NR_POLARIZATIONS; ++pol2) {
          const Visibility v = (*vis)[ch][bl][pol1][pol2];
          if (v.real() != 0.0f || v.imag() != 0.0f) {
            std::cout << "vis[" << ch << "][" << bl << "][" << pol1 << "]["
                      << pol2 << "] = (" << v.real() * scales[bl] << ", "
                      << v.imag() * scales[bl] << ") where scale is "
                      << scales[bl] << "\n";
          }
        }
      }
    }
  }
}

inline void print_nonzero_samples(const Samples *samps) {
  for (int ch = 0; ch < NR_CHANNELS; ++ch) {
    for (int j = 0; j < spatial::NR_BLOCKS_FOR_CORRELATION; ++j) {
      for (int k = 0; k < NR_RECEIVERS; k++) {
        for (int pol = 0; pol < NR_POLARIZATIONS; ++pol) {
          for (int t = 0; t < spatial::NR_TIMES_PER_BLOCK; ++t) {
            const Sample s = (*samps)[ch][j][k][pol][t];
            if (CAST_TO_FLOAT(s.real()) != 0.0f ||
                CAST_TO_FLOAT(s.imag()) != 0.0f) {
              std::cout << "samp[" << ch << "][" << j << "][" << k << "]["
                        << pol << "][" << t << "] = ("
                        << static_cast<int>(s.real()) << ", "
                        << static_cast<int>(s.imag()) << ")\n";
            }
          }
        }
      }
    }
  }
}

inline void print_nonzero_beams(const BeamformedData *data,
                                const int num_channels,
                                const int num_polarizations,
                                const int num_beams, const int num_time_steps) {
  for (int ch = 0; ch < num_channels; ++ch) {
    for (int pol = 0; pol < num_polarizations; ++pol) {
      for (int beam = 0; beam < num_beams; ++beam) {
        for (int step = 0; step < num_time_steps; ++step) {

          const std::complex<float> val = (*data)[ch][pol][beam][step];
          if (val.real() != 0.0f || val.imag() != 0.0f) {
            std::cout << "data[" << ch << "][" << pol << "][" << beam << "]["
                      << step << "] = " << val.real() << " + " << val.imag()
                      << "i\n";
          }
        }
      }
    }
  }
}

constexpr int NUM_ITERATIONS = 10;
constexpr int NUM_FRAMES_PER_ITERATION = 100;
constexpr int NR_TOTAL_FRAMES_PER_CHANNEL = 5 * NUM_FRAMES_PER_ITERATION;
constexpr int NR_FPGA_SOURCES = 1;
constexpr int NR_RECEIVERS_PER_PACKET = NR_RECEIVERS / NR_FPGA_SOURCES;
constexpr int NR_INPUT_BUFFERS = 50;
constexpr int NR_BETWEEN_SAMPLES = 64;
constexpr int MIN_FREQ_CHANNEL = 252; // this is assuming I know this...

// typedef Sample Packet[NR_RECEIVERS_PER_PACKET][NR_POLARIZATIONS]
//                      [NR_TIME_STEPS_PER_PACKET];
// using Tin = int8_t;
// using Tscale = int16_t;
constexpr int NR_TIMES_PER_PACKET = 64;
// constexpr int NR_ACTUAL_RECEIVERS = 20;
constexpr int COMPLEX = 2;

typedef Sample Packet[NR_TIME_STEPS_PER_PACKET][NR_RECEIVERS_PER_PACKET];
typedef Sample PacketSamples[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION]
                            [NR_TIME_STEPS_PER_PACKET][NR_ACTUAL_RECEIVERS];
typedef bool SampleOccupancy[NR_CHANNELS][spatial::NR_BLOCKS_FOR_CORRELATION]
                            [NR_RECEIVERS];
struct BufferState {
  bool is_ready;
  int start_seq;
  int end_seq;
  std::array<bool, NR_CHANNELS> is_populated{};
};

typedef PacketEntry Packets[RING_BUFFER_SIZE];

struct ProcessorStateBase {
public:
  int current_buffer = 0;
  std::atomic<int> write_index = 0;
  std::atomic<int> read_index = 0;
  std::array<BufferState, NR_INPUT_BUFFERS> buffers;
  uint64_t latest_packet_received[NR_CHANNELS][NR_FPGA_SOURCES] = {};
  std::vector<uint32_t> fpga_ids{};
  bool buffers_initialized = false;
  int running = 1;
  unsigned long long packets_received = 0;
  unsigned long long packets_processed = 0;
  virtual void *get_next_write_pointer() = 0;
  virtual void *get_current_write_pointer() = 0;
  virtual void add_received_packet_metadata(const int length,
                                            const sockaddr_in &client_addr) = 0;
};
template <typename T> struct ProcessorState : public ProcessorStateBase {
  // Public member variables
  typename T::PacketSamplesType *d_samples[NR_INPUT_BUFFERS];
  typename T::PacketEntryType *d_packet_data[RING_BUFFER_SIZE];

  // Constructor / Destructor
  ProcessorState() {

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
        d_samples[i] = (typename T::PacketSamplesType *)calloc(
            1, sizeof(typename T::PacketSamples));
        if (!d_samples[i])
          throw std::bad_alloc();
      }
    } catch (...) {
      cleanup();
      throw;
    }
  };
  ~ProcessorState() { cleanup(); };

  // Methods
  bool get_next_write_index() {
    int next_write_index =
        (write_index.load(std::memory_order_relaxed) + 1) % RING_BUFFER_SIZE;
    if (next_write_index == read_index.load(std::memory_order_acquire)) {
      printf("Ring buffer is full!! Dropping packets...\n");
      return false;
    }
    write_index.store(next_write_index, std::memory_order_release);
    // while (d_packet_data[write_index] != nullptr &&
    //        !d_packet_data[write_index]->processed) {
    //   write_index = (write_index + 1) % RING_BUFFER_SIZE;
    // }
    printf("Next write index is...%i ", next_write_index);
    return true;
  };
  void copy_data_to_input_buffer_if_able(ProcessedPacket &pkt) {

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
        std::cout << "Discarding packet as it is before current buffer "
                     "with begin_seq "
                  << buffers[current_buffer].start_seq
                  << " actually has packet_index " << pkt.sample_count
                  << std::endl;
        *pkt.original_packet_processed = true;
        break;
      }

      if (packet_index >= 0 && packet_index < NR_PACKETS_FOR_CORRELATION) {
        int receiver_index = fpga_index * NR_RECEIVERS_PER_PACKET;
        std::cout << "Copying data to packet_index " << packet_index
                  << " and channel index " << freq_channel
                  << " and receiver index " << receiver_index << " of buffer "
                  << (current_buffer + buffer_index) % NR_INPUT_BUFFERS
                  << std::endl;
        std::memcpy(&(*d_samples[buffer_index])[freq_channel][packet_index]
                                               [receiver_index],
                    // this is almost certainly not right.
                    pkt.payload->data, sizeof(PacketDataStructure));
        std::cout << "Setting original_packet_processed as true...\n";
        std::cout << "DEBUG: original_packet_processed_before="
                  << *pkt.original_packet_processed << std::endl;
        *(pkt.original_packet_processed) = true;
        std::cout << "DEBUG: original_packet_processed_after="
                  << *pkt.original_packet_processed << std::endl;
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
    printf("Processing packet: sample_count=%lu, freq_channel=%u, fpga_id=%u, "
           "payload=%d bytes\n",
           parsed.sample_count, parsed.freq_channel, parsed.fpga_id,
           parsed.payload_size);

    printf("First data point...%i + %i i\n", parsed.payload->data[0][0].real(),
           parsed.payload->data[0][0].imag());

    if (!buffers_initialized) {
      printf("Initializing buffers as this is the first packet...\n");
      initialize_buffers(parsed.sample_count);
      buffers_initialized = true;
    }
    // Simulate processing time
    copy_data_to_input_buffer_if_able(parsed);
    if (*parsed.original_packet_processed) {
      packets_processed += 1;
    }
  };
  void advance_to_next_buffer() {

    buffers[current_buffer].is_ready = false;

    buffers[current_buffer].is_ready = true;
    int old_buffer = current_buffer;
    int end_current_buffer_seq = buffers[old_buffer].end_seq;

    // Move to next buffer
    current_buffer = (current_buffer + 1) % NR_INPUT_BUFFERS;

    std::cout << "current_buffer is " << current_buffer << " and is it ready? "
              << buffers[current_buffer].is_ready << std::endl;

    // this is kinda assuming there's an async callback that will
    // ready the buffer after the data has been transferred out.
    while (!buffers[current_buffer].is_ready) {
      std::cout << "Waiting for buffer to be ready..." << std::endl;
    }

    // Reset new current buffer
    std::memset(std::begin(buffers[current_buffer].is_populated), (int)false,
                NR_CHANNELS);
    buffers[current_buffer].start_seq =
        end_current_buffer_seq + NR_BETWEEN_SAMPLES;
    buffers[current_buffer].end_seq =
        buffers[current_buffer].start_seq +
        (NR_PACKETS_FOR_CORRELATION - 1) * NR_BETWEEN_SAMPLES;

    // Update old buffer for future use
    int max_end_seq_in_buffers = 0;
    for (auto i = 0; i < NR_INPUT_BUFFERS; ++i) {
      max_end_seq_in_buffers =
          std::max(max_end_seq_in_buffers, buffers[i].end_seq);
    }
    buffers[old_buffer].start_seq =
        max_end_seq_in_buffers + 1 * NR_BETWEEN_SAMPLES;
    buffers[old_buffer].end_seq =
        buffers[old_buffer].start_seq +
        (NR_PACKETS_FOR_CORRELATION - 1) * NR_BETWEEN_SAMPLES;

    std::cout
        << "Current buffer is all complete. Moving to next buffer which is #"
        << current_buffer << std::endl;
    std::cout << "New buffer starts at packet "
              << buffers[current_buffer].start_seq << " and ends at "
              << buffers[current_buffer].end_seq << "\n";
  }

  void check_buffer_completion() {

    if (!buffers_initialized) {
      return;
    }

    for (auto channel = 0; channel < NR_CHANNELS; ++channel) {
      std::cout << "Check if buffers are complete for channel " << channel
                << std::endl;

      if (std::all_of(std::begin(latest_packet_received[channel]),
                      std::end(latest_packet_received[channel]), [this](int x) {
                        return x >= buffers[current_buffer].end_seq;
                      })) {
        buffers[current_buffer].is_populated[channel] = true;
        std::cout << "Buffer is complete for channel " << channel << ".\n";
      } else {
        std::cout << "Buffer is not complete for channel " << channel
                  << " as end_seq is " << buffers[current_buffer].end_seq
                  << " and latest_packet_receives are ";
        for (int check = 0; check < NR_FPGA_SOURCES; ++check) {
          std::cout << latest_packet_received[channel][check] << ", ";
        }
        std::cout << std::endl;
      }
    }
  };

  void write_buffer_to_hdf5(const int buffer_index,
                            const std::string &filename) {

    using namespace HighFive;

    // Create / overwrite file
    File file(filename, File::Overwrite);

    // Pointer to your samples
    PacketSamples *samples = d_samples[buffer_index];

    // Dataset shape
    std::vector<size_t> dims = {NR_CHANNELS, NR_PACKETS_FOR_CORRELATION,
                                NR_TIME_STEPS_PER_PACKET, NR_ACTUAL_RECEIVERS};

    // Create dataset of complex<int8_t> (or whatever Sample is)
    DataSet dataset =
        file.createDataSet<Sample>("packet_samples", DataSpace(dims));

    // Write buffer
    dataset.write(*samples);
  };
  void process_packets() {

    printf("Processor thread started\n");
    static bool first_written = false;
    int packets_processed_before_completion_check = 0;
    int current_read_index;
    while (running) {
      while (true) {
        current_read_index = read_index.load(std::memory_order_relaxed);

        if (current_read_index != write_index.load(std::memory_order_acquire)) {
          break;
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
        printf("Checking buffer completion...\n");
        check_buffer_completion();
        if (std::all_of(buffers[current_buffer].is_populated.begin(),
                        buffers[current_buffer].is_populated.end(),
                        [](bool i) { return i; })) {
          // Send off data to be processed by CUDA pipeline.
          // Then advance to next buffer and keep iterating.
          if (!first_written) {
            write_buffer_to_hdf5(current_buffer, "first_buffer.hdf5");
            first_written = true;
          }
          advance_to_next_buffer();
        }
        packets_processed_before_completion_check = 0;
      }
    }

    printf("Processor thread exiting\n");
  };

  void *get_current_write_pointer() {
    return (void *)&(d_packet_data[write_index]->data);
  }
  void *get_next_write_pointer() {
    while (!get_next_write_index()) {
      printf("Waiting for next pointer....\n");
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

  // Delete copy/move
  ProcessorState<T>(const ProcessorState<T> &) = delete;
  ProcessorState<T> &operator=(const ProcessorState<T> &) = delete;
  ProcessorState<T>(const ProcessorState<T> &&) = delete;
  ProcessorState<T> &operator=(ProcessorState<T> &&) = delete;

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

    printf("Receiver thread started\n");
    void *next_write_pointer = state.get_current_write_pointer();
    while (state.running) {
      int received = recvfrom(sockfd, next_write_pointer, buffer_size, 0,
                              (struct sockaddr *)&client_addr, &client_len);
      if (received < 0) {
        if (errno == EINTR)
          continue;
        perror("recvfrom");
        break;
      }

      state.add_received_packet_metadata(received, client_addr);
      state.packets_received += 1;
      next_write_pointer = state.get_next_write_pointer();
      // Store in ring buffer
      // store_packet(buffer, received, &client_addr, state);
    }

    printf("Receiver thread exiting\n");
  };

private:
  int sockfd;
  struct sockaddr_in server_addr;
  int port;
  int buffer_size;
};
