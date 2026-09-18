#pragma once
#include "spatial/logging.hpp"
#include <atomic>
#include <complex>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <netinet/in.h>
#include <sys/time.h>
#include <unistd.h>
#define BUFFER_SIZE 4096
#define MIN_PCAP_HEADER_SIZE 64

#pragma pack(push, 1)
struct EthernetHeader {
  uint8_t dst[6];
  uint8_t src[6];
  uint16_t ethertype;
};

struct IPHeader {
  uint8_t version_ihl;
  uint8_t dscp_ecn;
  uint16_t total_length;
  uint16_t identification;
  uint16_t flags_fragment;
  uint8_t ttl;
  uint8_t protocol;
  uint16_t header_checksum;
  uint32_t src_ip;
  uint32_t dst_ip;
};

struct UDPHeader {
  uint16_t src_port;
  uint16_t dst_port;
  uint16_t length;
  uint16_t checksum;
};

struct CustomHeader {
  uint64_t sample_count;
  uint32_t fpga_id;
  uint16_t freq_channel;
  uint8_t padding[8];
};

#pragma pack(pop)

struct FinalPacketData {
  uint64_t start_seq_id;
  uint64_t end_seq_id;
  size_t buffer_index;

  virtual void *get_samples_ptr() = 0;
  virtual size_t get_samples_elements_size() = 0;

  virtual void *get_scales_ptr() = 0;
  virtual size_t get_scales_element_size() = 0;

  virtual bool *get_arrivals_ptr() = 0;
  virtual size_t get_arrivals_size() = 0;

  virtual void zero_missing_packets() = 0;
  virtual int get_num_missing_packets() = 0;
};

// This one needs to be like this because it will be defined in the
// PacketStructure struct.
template <typename PacketSamplesType, typename PacketScalesType,
          size_t NR_CHANNELS, size_t NR_PACKETS_FOR_CORRELATION,
          size_t NR_RECEIVERS_PER_PACKET, size_t NR_POLARIZATIONS,
          size_t NR_FPGAS>
struct LambdaFinalPacketData : public FinalPacketData {
  using ArrivalsType =
      bool[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION + 2][NR_FPGAS];
  PacketSamplesType *samples = nullptr;
  PacketScalesType *scales = nullptr;
  ArrivalsType *arrivals = nullptr;

  void *get_samples_ptr() override { return (void *)samples; };
  void *get_scales_ptr() override { return (void *)scales; };
  bool *get_arrivals_ptr() override { return (bool *)arrivals; };

  size_t get_samples_elements_size() override {
    return sizeof(PacketSamplesType);
  };
  size_t get_scales_element_size() override {
    return sizeof(PacketScalesType);
  };

  size_t get_arrivals_size() override { return sizeof(ArrivalsType); };

  void zero_missing_packets() override {
    // we want to zero out missing packets including the extended ones.
    for (auto i = 0; i < NR_CHANNELS; ++i) {
      // be careful - need to convert the size_t to int otherwise signed ->
      // unsigned conversion takes place.
      for (auto j = -1; j < static_cast<int>(NR_PACKETS_FOR_CORRELATION) + 1;
           ++j) {
        for (auto k = 0; k < NR_FPGAS; ++k) {
          if (arrivals[0][i][j + 1][k] == 0) {
            for (auto m = 0; m < NR_RECEIVERS_PER_PACKET; ++m) {
              for (auto n = 0; n < NR_POLARIZATIONS; ++n) {
                scales[0][i][j + 1][k * NR_RECEIVERS_PER_PACKET + m][n] = 0;
              }
            }
          }
        }
      }
    }
  };
  int get_num_missing_packets() override {
    // we don't want to double-count missing packets so only include
    // from 0 -> NR_PACKETS_FOR_CORRELATION
    int sum = 0;
    for (auto i = 0; i < NR_CHANNELS; ++i) {
      for (auto j = 0; j < NR_PACKETS_FOR_CORRELATION; ++j) {
        for (auto k = 0; k < NR_FPGAS; ++k) {
          if (arrivals[0][i][j + 1][k] == 0) {
            sum++;
          }
        }
      }
    }
    return sum;
  };

  LambdaFinalPacketData() {

    // allocate samples
    CUDA_CHECK(cudaHostAlloc((void **)&samples, sizeof(PacketSamplesType),
                             cudaHostAllocDefault));
    std::memset(samples, 0, sizeof(PacketSamplesType));
    // allocate scales
    CUDA_CHECK(cudaHostAlloc((void **)&scales, sizeof(PacketScalesType),
                             cudaHostAllocDefault));
    std::memset(scales, 0, sizeof(PacketScalesType));
    CUDA_CHECK(cudaHostAlloc((void **)&arrivals, sizeof(ArrivalsType),
                             cudaHostAllocDefault));
    std::memset(arrivals, 0, sizeof(ArrivalsType));
  };
  ~LambdaFinalPacketData() {
    CUDA_CHECK(cudaFreeHost(samples));
    CUDA_CHECK(cudaFreeHost(scales));
    CUDA_CHECK(cudaFreeHost(arrivals));
  };
};

template <typename PacketScaleStructure, typename PacketDataStructure>
struct PacketPayload {
  PacketScaleStructure scales;
  PacketDataStructure data;
};

// Processed packet info
template <typename PacketScaleStructure, typename PacketDataStructure>
struct ProcessedPacket {
  uint64_t sample_count;
  uint64_t timestamp;
  const PacketPayload<PacketScaleStructure, PacketDataStructure> *payload;
  std::atomic<bool> *original_packet_processed;
  uint32_t fpga_id;
  uint32_t payload_size;
  uint16_t freq_channel;
} __attribute__((aligned(64)));

// Packet storage for ring buffer.
// Two-phase producer protocol: producer calls reserve_write_batch() (sets
// committed=false, advances write_index), fills slot->data[], then calls
// commit_write_batch() which sets committed=true (release) so the consumer
// sees the data.  committed=false while writing lets multiple NIC threads
// fill their reserved slots in parallel.
template <typename PacketScaleStructure, typename PacketDataStructure>
struct PacketEntry {
  // Slot capacity derived from the config instead of the fixed 4 KB
  // BUFFER_SIZE: largest frame this config can produce is a PCAP-replayed
  // Ethernet frame (42 B eth/ip/udp + CustomHeader + scales + samples).
  // Rounded up to a cache line.  For the default LAMBDA shape this shrinks
  // each slot from ~4.2 KB to ~2.7 KB, cutting the ring's working set by
  // ~35% (see scripts/profiling/ANALYSIS.md #3).
  static constexpr size_t DATA_CAPACITY =
      (42 + sizeof(CustomHeader) + sizeof(PacketScaleStructure) +
       sizeof(PacketDataStructure) + 63) &
      ~static_cast<size_t>(63);
  static_assert(DATA_CAPACITY >= MIN_PCAP_HEADER_SIZE,
                "slot must hold at least a minimal frame");
  uint8_t data[DATA_CAPACITY];
  int length;
  struct sockaddr_in sender_addr;
  struct timeval timestamp;
  // true  = consumer done, producer may claim (FREE)
  std::atomic<bool> processed{true};
  // true  = producer committed data, consumer may process (set with release)
  std::atomic<bool> committed{false};

  // std::atomic is neither copyable nor movable; provide an explicit move
  // constructor so PacketEntry can be returned by value in tests / helpers.
  PacketEntry() = default;
  PacketEntry(PacketEntry &&o) noexcept
      : length(o.length), sender_addr(o.sender_addr), timestamp(o.timestamp),
        processed(o.processed.load(std::memory_order_relaxed)),
        committed(o.committed.load(std::memory_order_relaxed)) {
    std::memcpy(data, o.data, sizeof(data));
  }

  virtual ProcessedPacket<PacketScaleStructure, PacketDataStructure>
  parse() = 0;
};

template <typename PacketScaleStructure, typename PacketDataStructure,
          bool OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET>
struct LambdaPacketEntry
    : public PacketEntry<PacketScaleStructure, PacketDataStructure> {
  using Base = PacketEntry<PacketScaleStructure, PacketDataStructure>;
  LambdaPacketEntry() = default;
  LambdaPacketEntry(LambdaPacketEntry &&o) noexcept : Base(std::move(o)) {}

  __attribute__((hot)) __attribute__((flatten))
  ProcessedPacket<PacketScaleStructure, PacketDataStructure>
  parse() noexcept override {

    const int length = this->length;
    const uint8_t *__restrict__ base = this->data;
    uint32_t offset = 0;
    if (length > sizeof(PacketScaleStructure) + sizeof(PacketDataStructure) +
                     sizeof(CustomHeader)) {
      offset = 42;
    }
    if (length < MIN_PCAP_HEADER_SIZE) [[unlikely]] {
      this->processed.store(true, std::memory_order_relaxed);
      return {};
    }

    const CustomHeader *__restrict__ custom =
        (const CustomHeader *)(base + offset);

    uint32_t fpga_id;
    if (OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET) {
      const uint8_t *ip_bytes =
          (const uint8_t *)&this->sender_addr.sin_addr.s_addr;
      uint8_t third_octet = ip_bytes[2];
      fpga_id = (uint32_t)third_octet;
    } else {
      fpga_id = custom->fpga_id;
    }

    return ProcessedPacket<PacketScaleStructure, PacketDataStructure>{
        .sample_count = custom->sample_count,
        .timestamp =
            this->timestamp.tv_sec * 1000000ULL + this->timestamp.tv_usec,
        .payload = reinterpret_cast<
            const PacketPayload<PacketScaleStructure, PacketDataStructure> *>(
            base + offset + sizeof(CustomHeader)),
        .original_packet_processed = &this->processed,
        // for now - take the IP address as the fpga_id.
        // i.e. 10.0.3.10 = FPGA ID 3.
        .fpga_id = fpga_id, // custom->fpga_id,
        .payload_size =
            static_cast<uint32_t>(length - (offset + sizeof(CustomHeader))),
        .freq_channel = custom->freq_channel};
  };
};

// NR_CHANNELS_T is the number of coarse channels delivered per-packet by the FPGAs -- kept as the
// first template parameter (unchanged position, for source compatibility with every existing call
// site) but its *meaning* is now "coarse/FPGA channel count", exposed as NR_FPGA_CHANNELS below.
// NR_FINE_CHANNELS_T (trailing, defaulted to 1) is the number of gpu-filter PFB fine channels each
// coarse channel is split into pre-correlation; NR_CHANNELS becomes the derived, widened
// "processing/output channel count" used by everything downstream of channelization
// (visibilities, beam output, eigendecomposition, spectra). With the default of 1, NR_CHANNELS ==
// NR_FPGA_CHANNELS and every type below is numerically identical to before this parameter existed.
//
// NR_FINE_CHANNEL_EDGE_TRIM_T (trailing, defaulted to 0) is how many fine channels to drop from
// *each* edge of every coarse channel's fine-channel breakdown before correlation/beamforming --
// the FPGA's own coarse channelizer is itself a 32/27-oversampled PFB (see
// scripts/fm_radio_run_1_alveo_0_eigenvector_directions.py's BW_PER_COARSE_CHANNEL), so raw samples
// near each coarse channel's edge carry that channelizer's own guard-band aliasing, and
// gpu-filter's fine channels nearest those edges inherit it. gpu-filter/FineChannelizer<T> itself
// still computes the full NR_FINE_CHANNELS_T (the PFB can't selectively skip edge channels); only
// NR_CHANNELS (and everything derived from it -- visibilities, beam output, eigendecomposition) is
// sized by the trimmed count, and channelizer_output_to_corr_input (spatial.cuh) is what actually
// drops the edge fine channels when gathering FineChannelizer's output. Default 0 keeps every
// existing (non-edge-trimmed) config numerically identical.
template <size_t NR_CHANNELS_T, size_t NR_FPGA_SOURCES_T,
          size_t NR_TIME_STEPS_PER_PACKET_T, size_t NR_RECEIVERS_T,
          size_t NR_POLARIZATIONS_T, size_t NR_RECEIVERS_PER_PACKET_T,
          size_t NR_PACKETS_FOR_CORRELATION_T, size_t NR_BEAMS_T,
          size_t NR_PADDED_RECEIVERS_T, size_t NR_PADDED_RECEIVERS_PER_BLOCK_T,
          size_t NR_CORRELATED_BLOCKS_TO_ACCUMULATE_T,
          bool OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET = false,
          size_t FFT_DOWNSAMPLE_FACTOR_T = 128, size_t NR_FINE_CHANNELS_T = 1,
          size_t NR_FINE_CHANNEL_EDGE_TRIM_T = 0>
struct LambdaConfig {

  static constexpr size_t NR_FPGA_CHANNELS = NR_CHANNELS_T;
  static constexpr size_t NR_FINE_CHANNELS = NR_FINE_CHANNELS_T;
  static constexpr size_t NR_FINE_CHANNEL_EDGE_TRIM = NR_FINE_CHANNEL_EDGE_TRIM_T;
  static_assert(NR_FINE_CHANNELS == 1 ||
                    NR_FINE_CHANNELS > 2 * NR_FINE_CHANNEL_EDGE_TRIM,
                "NR_FINE_CHANNEL_EDGE_TRIM_T trims more channels than NR_FINE_CHANNELS_T provides");
  // Number of fine channels per coarse channel that actually survive into NR_CHANNELS/correlation
  // after channelizer_output_to_corr_input drops NR_FINE_CHANNEL_EDGE_TRIM from each edge.
  static constexpr size_t NR_EFFECTIVE_FINE_CHANNELS =
      (NR_FINE_CHANNELS > 1) ? (NR_FINE_CHANNELS - 2 * NR_FINE_CHANNEL_EDGE_TRIM)
                             : NR_FINE_CHANNELS;
  // Widened processing/output channel count -- see the template-parameter comment above.
  static constexpr size_t NR_CHANNELS = NR_FPGA_CHANNELS * NR_EFFECTIVE_FINE_CHANNELS;
  static constexpr size_t NR_FPGA_SOURCES = NR_FPGA_SOURCES_T;
  static constexpr size_t NR_TIME_STEPS_PER_PACKET = NR_TIME_STEPS_PER_PACKET_T;
  static constexpr size_t NR_RECEIVERS = NR_RECEIVERS_T;
  static constexpr size_t NR_POLARIZATIONS = NR_POLARIZATIONS_T;
  static constexpr size_t NR_RECEIVERS_PER_PACKET = NR_RECEIVERS_PER_PACKET_T;
  static constexpr size_t NR_PACKETS_FOR_CORRELATION =
      NR_PACKETS_FOR_CORRELATION_T;
  static constexpr size_t NR_BEAMS = NR_BEAMS_T;
  static constexpr size_t NR_PADDED_RECEIVERS = NR_PADDED_RECEIVERS_T;
  static constexpr size_t NR_PADDED_RECEIVERS_PER_BLOCK =
      NR_PADDED_RECEIVERS_PER_BLOCK_T;
  static constexpr size_t NR_CORRELATED_BLOCKS_TO_ACCUMULATE =
      NR_CORRELATED_BLOCKS_TO_ACCUMULATE_T;
  static constexpr size_t NR_BASELINES =
      NR_PADDED_RECEIVERS * (NR_PADDED_RECEIVERS + 1) / 2;
  static constexpr size_t NR_BASELINES_UNPADDED =
      NR_RECEIVERS * (NR_RECEIVERS + 1) / 2;
  static constexpr size_t COMPLEX = 2;

  static constexpr size_t NR_TIME_STEPS_FOR_CORRELATION =
      NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET;

  // Post-channelization per-(fine-)channel time length: what BeamOutputType, the beamforming
  // GEMM, and every Output<T> implementation actually work with. Exact (no truncation) as long as
  // NR_TIME_STEPS_FOR_CORRELATION divides evenly by NR_FINE_CHANNELS, which FineChannelizer's own
  // static_asserts (pipeline/common.hpp) already enforce whenever NR_FINE_CHANNELS > 1. Equals
  // NR_TIME_STEPS_FOR_CORRELATION when NR_FINE_CHANNELS == 1.
  static constexpr size_t NR_TIME_STEPS_PER_FINE_CHANNEL =
      NR_TIME_STEPS_FOR_CORRELATION / NR_FINE_CHANNELS;

  // Packet-parsing/ingest types: shaped by however many coarse channels the FPGAs actually
  // deliver (NR_FPGA_CHANNELS), not the post-channelization processing count (NR_CHANNELS).
  template <typename T, int RECEIVERS = NR_RECEIVERS>
  using LambdaPacketSamplesT =
      std::complex<T>[NR_FPGA_CHANNELS][NR_PACKETS_FOR_CORRELATION + 2]
                     [NR_TIME_STEPS_PER_PACKET][RECEIVERS][NR_POLARIZATIONS];

  template <typename T, int RECEIVERS = NR_RECEIVERS>
  using LambdaPacketAlignedSamplesT =
      std::complex<T>[NR_FPGA_CHANNELS][NR_PACKETS_FOR_CORRELATION]
                     [NR_TIME_STEPS_PER_PACKET][RECEIVERS][NR_POLARIZATIONS];

  using PacketScalesType =
      int16_t[NR_FPGA_CHANNELS][NR_PACKETS_FOR_CORRELATION + 2][NR_RECEIVERS]
             [NR_POLARIZATIONS];

  using Sample = std::complex<int8_t>;
  using InputPacketSamplesType =
      std::complex<int8_t>[NR_FPGA_CHANNELS][NR_PACKETS_FOR_CORRELATION + 2]
                          [NR_FPGA_SOURCES][NR_TIME_STEPS_PER_PACKET]
                          [NR_RECEIVERS_PER_PACKET][NR_POLARIZATIONS];
  using InputPacketSamplesPlanarType =
      std::complex<int8_t>[NR_FPGA_CHANNELS][NR_PACKETS_FOR_CORRELATION + 2]
                          [NR_FPGA_SOURCES][NR_TIME_STEPS_PER_PACKET]
                          [NR_RECEIVERS_PER_PACKET][NR_POLARIZATIONS];

  using HalfInputPacketSamplesPlanarType =
      __half2[NR_FPGA_CHANNELS][NR_PACKETS_FOR_CORRELATION + 2][NR_FPGA_SOURCES]
             [NR_TIME_STEPS_PER_PACKET][NR_RECEIVERS_PER_PACKET]
             [NR_POLARIZATIONS];

  using PacketSamplesType = LambdaPacketSamplesT<int8_t>;
  using HalfPacketSamplesType = LambdaPacketSamplesT<__half>;
  using HalfPacketAlignedSamplesType = LambdaPacketAlignedSamplesT<__half>;

  using PaddedPacketSamplesType =
      LambdaPacketAlignedSamplesT<__half, NR_PADDED_RECEIVERS>;
  using PacketScaleStructure =
      int16_t[NR_RECEIVERS_PER_PACKET][NR_POLARIZATIONS];
  using PacketDataStructure =
      std::complex<int8_t>[NR_TIME_STEPS_PER_PACKET][NR_RECEIVERS_PER_PACKET]
                          [NR_POLARIZATIONS];
  using PacketPayloadType =
      PacketPayload<PacketScaleStructure, PacketDataStructure>;
  using ProcessedPacketType =
      ProcessedPacket<PacketScaleStructure, PacketDataStructure>;
  using PacketEntryType =
      LambdaPacketEntry<PacketScaleStructure, PacketDataStructure,
                        OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET>;
  using PacketFinalDataType =
      LambdaFinalPacketData<InputPacketSamplesType, PacketScalesType,
                            NR_FPGA_CHANNELS, NR_PACKETS_FOR_CORRELATION,
                            NR_RECEIVERS_PER_PACKET, NR_POLARIZATIONS,
                            NR_FPGA_SOURCES>;
  using BeamOutputType =
      __half[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS]
            [NR_TIME_STEPS_PER_FINE_CHANNEL][COMPLEX];
  using ArrivalsOutputType =
      bool[NR_FPGA_CHANNELS][NR_PACKETS_FOR_CORRELATION + 2][NR_FPGA_SOURCES];
  using VisibilitiesOutputType =
      float[NR_CHANNELS][NR_BASELINES_UNPADDED][NR_POLARIZATIONS]
           [NR_POLARIZATIONS][COMPLEX];
  using EigenvalueOutputType =
      float[NR_CHANNELS][NR_POLARIZATIONS][NR_POLARIZATIONS][NR_RECEIVERS];
  using EigenvectorOutputType =
      std::complex<float>[NR_CHANNELS][NR_POLARIZATIONS][NR_POLARIZATIONS]
                         [NR_RECEIVERS][NR_RECEIVERS];
  // Used only by LambdaAntennaSpectraPipeline (no beamforming there) -- sized by NR_RECEIVERS, see
  // the MultiChannelFFTCUFFT*Type comment below for why NR_BEAMS was wrong here.
  using FFTCUFFTPreprocessingType =
      __half2[NR_CHANNELS][NR_POLARIZATIONS][NR_RECEIVERS]
             [NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION];
  using FFTCUFFTInputType =
      float2[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS]
            [NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION];
  using FFTCUFFTOutputType =
      float2[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS]
            [NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION];

  // Antenna-level (not beamformed) FFT working buffers, used only by
  // LambdaAntennaSpectraPipeline -- sized by NR_RECEIVERS, not NR_BEAMS (there is no beamforming
  // in that pipeline). get_data_for_multi_channel_fft_launch/detect_and_downsample_multi_channel_fft_launch
  // (spatial.cuh) both size their total element count off NR_RECEIVERS explicitly; these types
  // previously declared NR_BEAMS here instead, a latent heap-buffer-overflow whenever
  // NR_BEAMS != NR_RECEIVERS (the common case -- NUMBER_BEAMS defaults to 1).
  using MultiChannelFFTCUFFTInputType =
      float2[NR_CHANNELS][NR_POLARIZATIONS][NR_RECEIVERS]
            [NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION];
  using MultiChannelFFTCUFFTOutputType =
      float2[NR_CHANNELS][NR_POLARIZATIONS][NR_RECEIVERS]
            [NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION];
  constexpr static int FFT_DOWNSAMPLE_FACTOR = FFT_DOWNSAMPLE_FACTOR_T;
  using FFTOutputType =
      float[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS]
           [NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION /
            FFT_DOWNSAMPLE_FACTOR];
  using AntennaFFTOutputType =
      float[NR_RECEIVERS][NR_TIME_STEPS_PER_PACKET *
                          NR_PACKETS_FOR_CORRELATION / FFT_DOWNSAMPLE_FACTOR];
  // LambdaAntennaSpectraPipeline's per-antenna spectral monitoring output. The trailing axis is
  // NR_TIME_STEPS_PER_FINE_CHANNEL (not the raw NR_TIME_STEPS_PER_PACKET*NR_PACKETS_FOR_CORRELATION
  // block length) so this naturally shrinks once channelized: gpu-filter's fine channels are
  // already the frequency decomposition (the widened NR_CHANNELS axis), so what's left per fine
  // channel is only its own time-domain samples, downsampled by simple power-averaging (no FFT --
  // see channelizer_output_to_antenna_power in spatial.cuh) rather than a second frequency
  // decomposition. Identical to today's shape when NR_FINE_CHANNELS == 1 (NR_TIME_STEPS_PER_FINE_CHANNEL
  // == NR_TIME_STEPS_FOR_CORRELATION then).
  using MultiChannelAntennaFFTOutputType =
      float[NR_CHANNELS][NR_POLARIZATIONS][NR_RECEIVERS]
           [NR_TIME_STEPS_PER_FINE_CHANNEL / FFT_DOWNSAMPLE_FACTOR];
  using Complex = std::complex<float>;
  using ReceiverArray = std::array<Complex, NR_RECEIVERS>;
  using PolArray = std::array<ReceiverArray, NR_POLARIZATIONS>;
  // Calibration gains are loaded per *coarse* channel from JSON and applied by
  // scale_and_convert_to_half before channelization -- see get_gains_structure() in common.hpp.
  using AntennaGains = std::array<PolArray, NR_FPGA_CHANNELS>;

  // Per-antenna delay in nanoseconds, one per receiver.
  using AntennaDelays = std::array<float, NR_RECEIVERS>;

  // Precomputed fine-channel phase correction table [chan][recv][fine_bin]. Backs the
  // pre-channelization apply_fine_delay_correction() path (NR_FINE_CHANNELS==1 configs only --
  // retired in favour of gpu-filter's native delay compensation once channelization is active).
  // Each entry is {cos(phi), sin(phi)} where phi = -2*pi*f_fine*tau.
  using FineDelayPhases =
      float2[NR_FPGA_CHANNELS][NR_RECEIVERS][NR_TIME_STEPS_FOR_CORRELATION];

  // Temporary float2 workspace for scatter/FFT/IFFT/gather [chan][recv][pol][bin].
  using FineDelayWorkspace =
      float2[NR_FPGA_CHANNELS][NR_RECEIVERS][NR_POLARIZATIONS]
            [NR_TIME_STEPS_FOR_CORRELATION];
};
