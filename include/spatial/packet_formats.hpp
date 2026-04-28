#pragma once
#include "spatial/logging.hpp"
#include <complex>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <iostream>
#include <netinet/in.h>
#include <sys/time.h>
#include <unistd.h>
#define BUFFER_SIZE 4096
#define MIN_PCAP_HEADER_SIZE 64

// this is a hack as we are not utilizing the fp16 in math
// we just need a type that is 16 bytes for storage
using fp16_t = uint16_t;

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

  virtual int *get_arrivals_ptr() = 0;
  virtual size_t get_arrivals_size() = 0;

  virtual void zero_samples() = 0;
  virtual void zero_arrivals() = 0;
  virtual size_t get_num_missing_packets() = 0;
};

// This one needs to be like this because it will be defined in the
// PacketStructure struct.
template <typename PacketSamplesType, size_t NR_CHANNELS,
          size_t NR_PACKETS_FOR_CORRELATION, size_t NR_RECEIVERS_PER_PACKET,
          size_t NR_POLARIZATIONS, size_t NR_FPGAS,
          size_t NR_TIME_STEPS_PER_PACKET>
struct LambdaFinalPacketData : public FinalPacketData {
  using ArrivalsType = int[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION][NR_FPGAS];
  PacketSamplesType *samples = nullptr;
  ArrivalsType *arrivals = nullptr;

  void *get_samples_ptr() override { return (void *)samples; };
  int *get_arrivals_ptr() override { return (int *)arrivals; };

  size_t get_samples_elements_size() override {
    return sizeof(PacketSamplesType);
  };

  size_t get_arrivals_size() override { return sizeof(ArrivalsType); };

  void zero_samples() override {
    std::memset(get_samples_ptr(), 0, get_samples_elements_size());
  };

  void zero_arrivals() override {

    std::memset(arrivals, 0, sizeof(ArrivalsType));
  }
  size_t get_num_missing_packets() override {
    size_t sum = 0;
    for (auto i = 0; i < NR_CHANNELS; ++i) {
      for (auto j = 0; j < NR_PACKETS_FOR_CORRELATION; ++j) {
        for (auto k = 0; k < NR_FPGAS; ++k) {
          sum += NR_TIME_STEPS_PER_PACKET - arrivals[0][i][j][k];
        }
      }
    }
    size_t missing_packets =
        sum / NR_TIME_STEPS_PER_PACKET + (sum % NR_TIME_STEPS_PER_PACKET != 0);
    return missing_packets;
  };

  LambdaFinalPacketData() {

    // allocate samples
    CUDA_CHECK(cudaHostAlloc((void **)&samples, sizeof(PacketSamplesType),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&arrivals, sizeof(ArrivalsType),
                             cudaHostAllocDefault));
  };
  ~LambdaFinalPacketData() {
    cudaFreeHost(samples);
    cudaFreeHost(arrivals);
  };
};

template <typename PacketScaleStructure, typename PacketDataStructure>
struct PacketPayload {
  PacketScaleStructure scales;
  PacketDataStructure data;
};

// Processed packet info
template <typename PacketDataStructure> struct ProcessedPacket {
  uint64_t sample_count;
  uint64_t timestamp;
  const PacketDataStructure *payload;
  bool *original_packet_processed;
  uint32_t fpga_id;
  uint32_t payload_size;
  uint16_t freq_channel;
} __attribute__((aligned(64)));

// Packet storage for ring buffer
template <typename PacketDataStructure> struct PacketEntry {
  uint8_t data[BUFFER_SIZE];
  int length;
  struct sockaddr_in sender_addr;
  struct timeval timestamp;
  bool processed; // 0 = unprocessed, 1 = processed

  virtual ProcessedPacket<PacketDataStructure> parse() = 0;
};

template <typename PacketScaleStructure, typename InputPacketDataStructure,
          typename OutputPacketDataStructure, size_t NR_RECEIVERS_PER_PACKET,
          size_t NR_POLARIZATIONS, size_t NR_TIME_STEPS_PER_PACKET,
          bool OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET>
struct LambdaPacketEntry : public PacketEntry<OutputPacketDataStructure> {
  __attribute__((hot))
  __attribute__((flatten)) ProcessedPacket<OutputPacketDataStructure>
  parse() noexcept override {

    // LOG_DEBUG("Entering parser...\n");
    const int length = this->length;
    const uint8_t *__restrict__ base = this->data;
    uint32_t offset = 0;
    if (length > sizeof(PacketScaleStructure) +
                     sizeof(InputPacketDataStructure) + sizeof(CustomHeader)) {
      offset = 42;
    }
    __builtin_prefetch(base + offset, 0, 3);
    if (length < MIN_PCAP_HEADER_SIZE) [[unlikely]] {
      this->processed = (length < MIN_PCAP_HEADER_SIZE);
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

    unpack_packet_data(
        reinterpret_cast<const PacketPayload<PacketScaleStructure,
                                             InputPacketDataStructure> *>(
            base + offset + sizeof(CustomHeader)));

    return ProcessedPacket<OutputPacketDataStructure>{
        .sample_count = custom->sample_count,
        .timestamp =
            this->timestamp.tv_sec * 1000000ULL + this->timestamp.tv_usec,
        .payload = &this->output_data,
        .original_packet_processed = &this->processed,
        // for now - take the IP address as the fpga_id.
        // i.e. 10.0.3.10 = FPGA ID 3.
        .fpga_id = fpga_id, // custom->fpga_id,
        .payload_size =
            static_cast<uint32_t>(length - (offset + sizeof(CustomHeader))),
        .freq_channel = custom->freq_channel};
  };

  OutputPacketDataStructure output_data;
  inline std::complex<int> convert_and_scale(std::complex<int8_t> z,
                                             int scale) {
    return {static_cast<int>(z.real()) * scale,
            static_cast<int>(z.imag()) * scale};
  };

  std::complex<__half> convert_float_to_half(std::complex<int> z) {
    return {__float2half(static_cast<float>(z.real())),
            __float2half(static_cast<float>(z.imag()))};
  };

  [[gnu::always_inline]] static inline void process_8_complex(
      const int8_t *__restrict__ src,     // 16 bytes: 8 × complex<int8>
      const int32_t *__restrict__ scales, // 8 × int32 scale factors
      fp16_t *__restrict__ dst            // 8 × complex<fp16> = 32 bytes output
  ) {
    // Load 16 bytes: 8 complex<int8_t>
    // [r0,i0,r1,i1,r2,i2,r3,i3, r4,i4,r5,i5,r6,i6,r7,i7]
    __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));

    // Deinterleave with pshufb: pull out all real bytes, then all imag bytes.
    // -1 (0x80) in the shuffle mask zeroes that output byte.
    const __m128i shuf_r =
        _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m128i shuf_i =
        _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 15, 13, 11, 9, 7, 5, 3, 1);
    __m128i reals_i8 = _mm_shuffle_epi8(raw, shuf_r); // [r0..r7] in low 8 bytes
    __m128i imags_i8 = _mm_shuffle_epi8(raw, shuf_i); // [i0..i7] in low 8 bytes

    // Sign-extend int8 → int32: _mm256_cvtepi8_epi32 reads the low 8 bytes of
    // its __m128i argument and produces 8 × int32 in a 256-bit register.
    __m256i reals_i32 = _mm256_cvtepi8_epi32(reals_i8);
    __m256i imags_i32 = _mm256_cvtepi8_epi32(imags_i8);

    // Load 8 scale factors and broadcast-multiply.
    // mullo_epi32 keeps the low 32 bits of each 32×32 product.
    // Scale values are already int32 so no overflow risk for typical radio
    // data.
    __m256i vscale =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(scales));
    __m256i scaled_r = _mm256_mullo_epi32(reals_i32, vscale);
    __m256i scaled_i = _mm256_mullo_epi32(imags_i32, vscale);

    // Convert int32 → float32 (exact for 32-bit integers up to 2^24)
    __m256 float_r = _mm256_cvtepi32_ps(scaled_r);
    __m256 float_i = _mm256_cvtepi32_ps(scaled_i);

    // Convert float32 → float16 using F16C (vcvtps2ph).
    // _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC = round-to-nearest, no
    // exceptions. Result: 8 × fp16 packed into a 128-bit register (8 × 16-bit
    // lanes).
    __m128i half_r =
        _mm256_cvtps_ph(float_r, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m128i half_i =
        _mm256_cvtps_ph(float_i, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Re-interleave: punpcklwd / punpckhwd merge two int16 streams
    // element-by-element. half_r = [r0,r1,r2,r3, r4,r5,r6,r7] half_i =
    // [i0,i1,i2,i3, i4,i5,i6,i7] lo = [r0,i0, r1,i1, r2,i2, r3,i3]
    // (complex<fp16> × 4 = 16 bytes) hi = [r4,i4, r5,i5, r6,i6, r7,i7]
    // (complex<fp16> × 4 = 16 bytes)
    __m128i lo = _mm_unpacklo_epi16(half_r, half_i);
    __m128i hi = _mm_unpackhi_epi16(half_r, half_i);

    // Store 32 bytes = 8 × complex<fp16>
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), lo);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 8), hi);
  }

  [[gnu::always_inline]] static inline void process_4_complex(
      const int8_t *__restrict__ src,     // 8 bytes: 4 × complex<int8>
      const int32_t *__restrict__ scales, // 4 × int32
      fp16_t *__restrict__ dst            // 4 × complex<fp16> = 16 bytes
  ) {
    // _mm_loadl_epi64: loads exactly 8 bytes into the low half of a __m128i,
    // zeroing the upper half. Avoids any read past the end of the buffer.
    __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src));

    const __m128i shuf_r = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                        -1, -1, 6, 4, 2, 0);
    const __m128i shuf_i = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                        -1, -1, 7, 5, 3, 1);
    __m128i reals_i8 = _mm_shuffle_epi8(raw, shuf_r);
    __m128i imags_i8 = _mm_shuffle_epi8(raw, shuf_i);

    // _mm_cvtepi8_epi32: SSE4.1 scalar of the AVX2 version — sign-extends
    // the low 4 bytes of the __m128i into 4 × int32.
    __m128i reals_i32 = _mm_cvtepi8_epi32(reals_i8);
    __m128i imags_i32 = _mm_cvtepi8_epi32(imags_i8);

    // 4 scales fit exactly in one __m128i (4 × 32-bit = 16 bytes).
    __m128i vscale = _mm_loadu_si128(reinterpret_cast<const __m128i *>(scales));
    __m128i scaled_r = _mm_mullo_epi32(reals_i32, vscale);
    __m128i scaled_i = _mm_mullo_epi32(imags_i32, vscale);

    __m128 float_r = _mm_cvtepi32_ps(scaled_r);
    __m128 float_i = _mm_cvtepi32_ps(scaled_i);

    // _mm_cvtps_ph: the 128-bit F16C variant. Converts 4 × float32 → 4 × fp16,
    // packed into the low 64 bits of the returned __m128i (upper 64 bits
    // zeroed).
    __m128i half_r =
        _mm_cvtps_ph(float_r, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m128i half_i =
        _mm_cvtps_ph(float_i, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Both registers have 4 fp16 values in their low 64 bits.
    // unpacklo_epi16 interleaves those low halves: [r0,i0, r1,i1, r2,i2,
    // r3,i3].
    __m128i interleaved = _mm_unpacklo_epi16(half_r, half_i);

    // Store exactly 16 bytes — 4 × complex<fp16>. No overrun.
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), interleaved);
  }

  void unpack_packet_data(
      const PacketPayload<PacketScaleStructure, InputPacketDataStructure>
          *payload) {
    constexpr int NR_CHANNELS = NR_RECEIVERS_PER_PACKET * NR_POLARIZATIONS;
    constexpr int FULL8 = NR_CHANNELS / 8; // how many AVX2 (8-wide) chunks
    constexpr int REM = NR_CHANNELS % 8;   // leftover channels after AVX2
    constexpr int HAS4 = (REM >= 4) ? 1 : 0;
    constexpr int SCALAR_TAIL = REM - HAS4 * 4; // 0–3 channels handled scalar

    // Offset of the 4-wide chunk (in channels), and scalar tail (in channels)
    constexpr int OFF_4 = FULL8 * 8;
    constexpr int OFF_SCALAR = FULL8 * 8 + HAS4 * 4;

    using clock = std::chrono::steady_clock;
    auto start = clock::now();

    alignas(32) int32_t scales[NR_CHANNELS] = {};
    for (int i = 0; i < NR_RECEIVERS_PER_PACKET; ++i)
      for (int j = 0; j < NR_POLARIZATIONS; ++j)
        scales[i * NR_POLARIZATIONS + j] =
            static_cast<int32_t>(payload->scales[i][j]);
    // scales[20..23] remain zero — harmless for the padded tail.

    // ── Step 2: Outer loop over time steps ─────────────────────────────────
    // data[k][0..9][0..1] is a contiguous 40-byte row in memory.
    // output_data[k][0..9][0..1] is a contiguous 80-byte row.
    // Processing k as the outermost dimension gives us sequential 40-byte
    // reads.
    for (int k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {

      const int8_t *__restrict__ src =
          reinterpret_cast<const int8_t *>(&payload->data[k][0][0]);
      fp16_t *__restrict__ dst =
          reinterpret_cast<fp16_t *>(&this->output_data[k][0][0]);

      // ── AVX2 (8-wide) chunks ────────────────────────────────────────────
      // Loop bound is a compile-time constant, so this is fully unrolled.
      // For your 10×2=20 case: FULL8=2, emits exactly 2 inlined call bodies.
      for (int c = 0; c < FULL8; ++c)
        process_8_complex(src + c * 16,   // 8 complex<int8> = 16 bytes
                          scales + c * 8, // 8 × int32
                          dst + c * 16    // 8 complex<fp16> = 16 fp16 values
        );

      // ── SSE (4-wide) chunk, if REM >= 4 ────────────────────────────────
      // if constexpr: branch is eliminated entirely at compile time when false.
      // For 10×2=20: REM=4, HAS4=1 — this branch is kept, OFF_4=16.
      // For  4×2= 8: REM=0, HAS4=0 — this branch disappears.
      if constexpr (HAS4) {
        process_4_complex(src + OFF_4 * 2, // byte offset
                          scales + OFF_4,
                          dst + OFF_4 * 2 // fp16 offset
        );
      }
      // ── Step 3: SIMD over all 20 channels, 8 at a time ─────────────────
      // 20 channels → chunk 0: [0..7], chunk 1: [8..15], chunk 2: [16..23]
      // Chunk 2 reads 4 real channels + 4 zero-padded; output is discarded for
      // [20..23]. Each complex<int8_t> = 2 bytes, complex<fp16> = 4 bytes.
    }

    // std::array<int, NR_POLARIZATIONS * NR_RECEIVERS_PER_PACKET> scales;
    // for (auto i = 0; i < NR_RECEIVERS_PER_PACKET; ++i) {
    //   for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
    //     scales[i * NR_POLARIZATIONS + j] =
    //         static_cast<int>(payload->scales[i][j]);
    //   }
    // }
    //
    // for (auto k = 0; k < NR_TIME_STEPS_PER_PACKET; ++k) {
    //   for (auto i = 0; i < NR_RECEIVERS_PER_PACKET; ++i) {
    //     for (auto j = 0; j < NR_POLARIZATIONS; ++j) {
    //
    //       std::complex<int8_t> data = payload->data[k][i][j];
    //       std::complex<int> data_int =
    //           convert_and_scale(data, scales[i * NR_POLARIZATIONS + j]);
    //       this->output_data[k][i][j] = convert_float_to_half(data_int);
    //     }
    //   }
    // }
    //
    auto end = clock::now();
    auto duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    LOG_INFO("Duration was {}ns", duration_ns);
  };
};

template <size_t NR_CHANNELS_T, size_t NR_FPGA_SOURCES_T,
          size_t NR_TIME_STEPS_PER_PACKET_T, size_t NR_RECEIVERS_T,
          size_t NR_POLARIZATIONS_T, size_t NR_RECEIVERS_PER_PACKET_T,
          size_t NR_PACKETS_FOR_CORRELATION_T, size_t NR_BEAMS_T,
          size_t NR_PADDED_RECEIVERS_T, size_t NR_PADDED_RECEIVERS_PER_BLOCK_T,
          size_t NR_CORRELATED_BLOCKS_TO_ACCUMULATE_T,
          bool OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET = false,
          size_t FFT_DOWNSAMPLE_FACTOR_T = 64>
struct LambdaConfig {

  static constexpr size_t NR_CHANNELS = NR_CHANNELS_T;
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

  template <typename T, int RECEIVERS = NR_RECEIVERS>
  using LambdaPacketSamplesT =
      std::complex<T>[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION]
                     [NR_TIME_STEPS_PER_PACKET][RECEIVERS][NR_POLARIZATIONS];

  template <typename T>
  using LambdaInputPacketSamplesPlanarT =
      T[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION][NR_FPGA_SOURCES]
       [NR_TIME_STEPS_PER_PACKET][NR_RECEIVERS_PER_PACKET][NR_POLARIZATIONS][2];
  template <typename T, int RECEIVERS = NR_RECEIVERS>
  using LambdaPacketSamplesPlanarT =
      T[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION][NR_TIME_STEPS_PER_PACKET]
       [RECEIVERS][NR_POLARIZATIONS][2]; // for complex

  using PacketScalesType = int16_t[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION]
                                  [NR_RECEIVERS][NR_POLARIZATIONS];

  using Sample = std::complex<int8_t>;
  using InputPacketSamplesType =
      std::complex<__half>[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION]
                          [NR_FPGA_SOURCES][NR_TIME_STEPS_PER_PACKET]
                          [NR_RECEIVERS_PER_PACKET][NR_POLARIZATIONS];
  using InputPacketSamplesPlanarType = LambdaInputPacketSamplesPlanarT<__half>;
  using PacketSamplesType = LambdaPacketSamplesT<int8_t>;
  using HalfPacketSamplesType = LambdaPacketSamplesT<__half>;
  using HalfInputPacketSamplesPlanarType =
      LambdaInputPacketSamplesPlanarT<__half>;

  using PaddedPacketSamplesType =
      LambdaPacketSamplesT<__half, NR_PADDED_RECEIVERS>;
  using PacketSamplesPlanarType = LambdaPacketSamplesPlanarT<int8_t>;
  using HalfPacketSamplesPlanarType = LambdaPacketSamplesPlanarT<__half>;
  using PaddedPacketSamplesPlanarType =
      LambdaPacketSamplesPlanarT<__half, NR_PADDED_RECEIVERS>;
  using PacketScaleStructure =
      int16_t[NR_RECEIVERS_PER_PACKET][NR_POLARIZATIONS];
  using PacketDataStructure =
      std::complex<int8_t>[NR_TIME_STEPS_PER_PACKET][NR_RECEIVERS_PER_PACKET]
                          [NR_POLARIZATIONS];
  using OutputPacketDataStructure =
      std::complex<__half>[NR_TIME_STEPS_PER_PACKET][NR_RECEIVERS_PER_PACKET]
                          [NR_POLARIZATIONS];
  using PacketPayloadType =
      PacketPayload<PacketScaleStructure, PacketDataStructure>;
  using ProcessedPacketType = ProcessedPacket<OutputPacketDataStructure>;
  using PacketEntryType =
      LambdaPacketEntry<PacketScaleStructure, PacketDataStructure,
                        OutputPacketDataStructure, NR_RECEIVERS_PER_PACKET,
                        NR_POLARIZATIONS, NR_TIME_STEPS_PER_PACKET,
                        OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET>;
  using PacketFinalDataType =
      LambdaFinalPacketData<InputPacketSamplesType, NR_CHANNELS,
                            NR_PACKETS_FOR_CORRELATION, NR_RECEIVERS_PER_PACKET,
                            NR_POLARIZATIONS, NR_FPGA_SOURCES,
                            NR_TIME_STEPS_PER_PACKET>;
  using BeamOutputType =
      __half[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS]
            [NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET][COMPLEX];
  using ArrivalsOutputType =
      bool[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION][NR_FPGA_SOURCES];
  using VisibilitiesOutputType =
      float[NR_CHANNELS][NR_BASELINES_UNPADDED][NR_POLARIZATIONS]
           [NR_POLARIZATIONS][COMPLEX];
  using EigenvalueOutputType =
      float[NR_CHANNELS][NR_POLARIZATIONS][NR_POLARIZATIONS][NR_RECEIVERS];
  using EigenvectorOutputType =
      std::complex<float>[NR_CHANNELS][NR_POLARIZATIONS][NR_POLARIZATIONS]
                         [NR_RECEIVERS][NR_RECEIVERS];
  using FFTCUFFTPreprocessingType =
      __half2[NR_CHANNELS][NR_POLARIZATIONS][NR_RECEIVERS]
             [NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION];
  using FFTCUFFTInputType = float2[NR_RECEIVERS][NR_TIME_STEPS_PER_PACKET *
                                                 NR_PACKETS_FOR_CORRELATION];
  using FFTCUFFTOutputType = float2[NR_RECEIVERS][NR_TIME_STEPS_PER_PACKET *
                                                  NR_PACKETS_FOR_CORRELATION];

  using MultiChannelFFTCUFFTInputType =
      float2[NR_CHANNELS][NR_POLARIZATIONS][NR_RECEIVERS]
            [NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION];
  using MultiChannelFFTCUFFTOutputType =
      float2[NR_CHANNELS][NR_POLARIZATIONS][NR_RECEIVERS]
            [NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION];
  constexpr static int FFT_DOWNSAMPLE_FACTOR = FFT_DOWNSAMPLE_FACTOR_T;
  using FFTOutputType =
      float[NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION /
            FFT_DOWNSAMPLE_FACTOR];
  using AntennaFFTOutputType =
      float[NR_RECEIVERS][NR_TIME_STEPS_PER_PACKET *
                          NR_PACKETS_FOR_CORRELATION / FFT_DOWNSAMPLE_FACTOR];
  using MultiChannelAntennaFFTOutputType =
      float[NR_CHANNELS][NR_POLARIZATIONS][NR_RECEIVERS]
           [NR_TIME_STEPS_PER_PACKET * NR_PACKETS_FOR_CORRELATION /
            FFT_DOWNSAMPLE_FACTOR];
  using PulsarFoldOutputType = float[NR_CHANNELS][16][NR_POLARIZATIONS][256];
};
