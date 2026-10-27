#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <endian.h>
#include <limits>
#include <stdexcept>

// LAMBDA CODIF: little-endian header, 64 time samples of 20 complex int8 ADCs.
namespace codif {
constexpr size_t header_bytes = 64;
constexpr size_t data_bytes = 2560;
constexpr uint64_t frames_per_period = 390625;
constexpr uint64_t samples_per_period = frames_per_period * 64;
constexpr uint32_t period_seconds = 27;
constexpr uint16_t metadata_id = 0x03bb;

inline uint16_t u16(const uint8_t *p) {
  uint16_t v; std::memcpy(&v, p, sizeof(v)); return le16toh(v);
}
inline uint32_t u32(const uint8_t *p) {
  uint32_t v; std::memcpy(&v, p, sizeof(v)); return le32toh(v);
}
inline uint64_t u64(const uint8_t *p) {
  uint64_t v; std::memcpy(&v, p, sizeof(v)); return le64toh(v);
}
inline bool has_sync(const uint8_t *p, size_t size) {
  return size >= header_bytes && u32(p + 40) == 0xfeedcafe;
}
inline bool valid(const uint8_t *p, size_t size) {
  // LAMBDA specifies the FRAME count here; also accept the CODIF sample count.
  const auto periods = size >= header_bytes ? u64(p + 32) : 0;
  return size >= header_bytes + data_bytes && has_sync(p, size) &&
         p[9] == 8 && (p[10] & 0xf7) == 4 && u16(p + 14) == period_seconds &&
         u16(p + 16) < 512 &&
         // Metadata ID zero is used by the original firmware simulation.
         (u16(p + 44) == metadata_id || u16(p + 44) == 0) &&
         u16(p + 24) == 20 && u16(p + 26) == 5 && u32(p + 28) == 320 &&
         (periods == frames_per_period || periods == samples_per_period) &&
         u32(p) < frames_per_period;
}

inline uint64_t epoch_seconds(uint8_t epoch) {
  const unsigned years = epoch / 2;
  const unsigned year = 2000 + years;
  const bool leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
  const uint64_t days = 365 * years + (years + 3) / 4 - (years > 100) +
                        (epoch % 2 ? 181 + leap : 0);
  return days * 86400;
}

inline uint64_t sample_count(const uint8_t *p) {
  // An absolute sample axis keeps streams aligned across frame/epoch rollover.
  const uint64_t seconds = epoch_seconds(p[8]) + u32(p + 4);
  return seconds * samples_per_period / period_seconds + u32(p) * 64ULL;
}

inline void advance(uint8_t *p, uint64_t samples) {
  if (samples % 64)
    throw std::runtime_error("CODIF replay offset must be a multiple of 64 samples");
  const uint64_t frames = u32(p) + samples / 64;
  const uint64_t seconds = u32(p + 4) + frames / frames_per_period * period_seconds;
  if (seconds > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("CODIF replay epoch offset overflow");
  const uint32_t frame_le = htole32(frames % frames_per_period);
  const uint32_t seconds_le = htole32(seconds);
  std::memcpy(p, &frame_le, sizeof(frame_le));
  std::memcpy(p + 4, &seconds_le, sizeof(seconds_le));
}
} // namespace codif
