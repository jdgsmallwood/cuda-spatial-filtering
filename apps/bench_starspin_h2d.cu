// Starspin's actual samples + scales transfer sizes, with pinned host buffers.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <vector>

static void check(cudaError_t err) {
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

__global__ void copy_mapped_host(char *destination, const char *source, size_t bytes) {
  const size_t thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = thread; i < bytes / sizeof(uint4); i += stride)
    reinterpret_cast<uint4 *>(destination)[i] =
        reinterpret_cast<const uint4 *>(source)[i];
  for (size_t i = bytes / sizeof(uint4) * sizeof(uint4) + thread; i < bytes; i += stride)
    destination[i] = source[i];
}

__global__ void validate_copy(const unsigned char *data, size_t samples, size_t scales,
                             unsigned *errors) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < samples + scales; i += size_t(gridDim.x) * blockDim.x)
    if (data[i] != (i < samples ? 1 : 0)) atomicExch(errors, 1U);
}

// Report actual Linux backing rather than treating MADV_HUGEPAGE as proof.
// Counts describe VMAs overlapping the allocation (CUDA may use a larger VMA).
static void report_pages(void *ptr, size_t bytes, int buffer, const char *stage) {
  const auto first = reinterpret_cast<uintptr_t>(ptr), last = first + bytes;
  std::ifstream smaps("/proc/self/smaps");
  if (!smaps) throw std::runtime_error("cannot inspect /proc/self/smaps");
  std::set<uintptr_t> starts;
  std::string line;
  bool overlaps = false;
  size_t huge_kib = 0, rss_kib = 0;
  while (std::getline(smaps, line)) {
    unsigned long begin, end;
    if (std::sscanf(line.c_str(), "%lx-%lx", &begin, &end) == 2) {
      overlaps = begin < last && end > first;
      if (overlaps) starts.insert(begin);
    } else if (overlaps) {
      size_t value;
      if (std::sscanf(line.c_str(), "AnonHugePages: %zu kB", &value) == 1) huge_kib += value;
      if (std::sscanf(line.c_str(), "Rss: %zu kB", &value) == 1) rss_kib += value;
    }
  }
  std::cout << "host_pages stage=" << stage << " buffer=" << buffer
            << " address=" << ptr << " bytes=" << bytes
            << " overlapping_vma_anon_huge_kib=" << huge_kib
            << " overlapping_vma_rss_kib=" << rss_kib << '\n';
  std::ifstream numa("/proc/self/numa_maps");
  while (std::getline(numa, line)) {
    unsigned long begin;
    if (std::sscanf(line.c_str(), "%lx", &begin) == 1 && starts.count(begin))
      std::cout << "host_numa stage=" << stage << " buffer=" << buffer << ' ' << line << '\n';
  }
}

int main(int argc, char **argv) {
  try {
    const std::string size_spec = argc > 1 ? argv[1] : "54";
    const bool raw_bytes = size_spec.rfind("bytes:", 0) == 0;
    const int channels = raw_bytes ? 0 : std::atoi(size_spec.c_str());
    const size_t requested_bytes = raw_bytes ? std::stoull(size_spec.substr(6)) : 0;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 128;
    const bool combined = argc > 3 && std::strcmp(argv[3], "combined") == 0;
    const int stream_count = argc > 4 ? std::atoi(argv[4]) : 3;
    const bool write_combined = argc > 5 && std::strcmp(argv[5], "wc") == 0;
    const bool registered = argc > 5 &&
        (std::strcmp(argv[5], "register") == 0 ||
         std::strcmp(argv[5], "register_no_thp") == 0);
    const bool no_thp = argc > 5 && std::strcmp(argv[5], "register_no_thp") == 0;
    const int buffers = argc > 6 ? std::atoi(argv[6]) : 3;
    const bool d2h = argc > 7 && std::strcmp(argv[7], "d2h") == 0;
    const size_t chunk_bytes = argc > 8 ? std::stoull(argv[8]) : 0;
    const int kernel_blocks = argc > 9 ? std::atoi(argv[9]) : 0;
    const bool rounded = argc > 10 && std::strcmp(argv[10], "rounded") == 0;
    const int dma_percent = argc > 11 ? std::atoi(argv[11]) : 0;
    if ((!raw_bytes && (channels < 1 || channels > 80)) ||
        (raw_bytes && (requested_bytes < 1 || requested_bytes > 14000000000ULL)) ||
        repeats < 1 || repeats > 1000 || stream_count < 1 || stream_count > 8 ||
        buffers < 1 || buffers > 8 || stream_count > buffers ||
        (raw_bytes && !combined) || (registered && !combined) ||
        kernel_blocks < 0 || kernel_blocks > 4096 ||
        (kernel_blocks && (d2h || !combined || chunk_bytes)) || (rounded && !registered) ||
        dma_percent < 0 || dma_percent >= 100 || (dma_percent && !kernel_blocks))
      throw std::runtime_error("usage: bench_starspin_h2d [channels 1..80|bytes:N<=14000000000] [repeats 1..1000] [split|combined] [streams 1..8] [default|wc|register|register_no_thp] [buffers 1..8] [h2d|d2h] [chunk-bytes: 0=whole] [kernel-blocks: 0=DMA] [exact|rounded] [DMA-prefix-percent: 0=kernel-only]");
    constexpr size_t packets = 256 + 2, fpgas = 4, receivers = 40, pols = 2;
    constexpr size_t samples_per_packet = 64;
    const size_t samples = raw_bytes ? requested_bytes :
        size_t(channels) * packets * fpgas * samples_per_packet *
        (receivers / fpgas) * pols * 2;
    const size_t scales = raw_bytes ? 0 :
        size_t(channels) * packets * receivers * pols * 2;
    const unsigned host_flags = write_combined ? cudaHostAllocWriteCombined
                                                : cudaHostAllocDefault;
    constexpr size_t alignment = 2 * 1024 * 1024;
    const size_t registered_bytes = rounded ?
        (samples + scales + alignment - 1) / alignment * alignment : samples + scales;
    std::vector<void *> host_samples(buffers), host_scales(buffers);
    std::vector<void *> device_samples(buffers), device_scales(buffers);
    std::vector<void *> mapped_base(buffers, nullptr);
    std::vector<void *> mapped_device(buffers, nullptr);
    std::vector<cudaStream_t> streams(buffers);
    std::vector<cudaStream_t> kernel_streams(buffers);
    for (int i = 0; i < buffers; ++i) {
      if (registered) {
        const size_t alloc_bytes = registered_bytes + alignment;
        mapped_base[i] = mmap(nullptr, alloc_bytes, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped_base[i] == MAP_FAILED) throw std::runtime_error("mmap failed");
        host_samples[i] = reinterpret_cast<void *>(
            (reinterpret_cast<uintptr_t>(mapped_base[i]) + alignment - 1) &
            ~(uintptr_t(alignment) - 1));
        if (madvise(host_samples[i], registered_bytes,
                    no_thp ? MADV_NOHUGEPAGE : MADV_HUGEPAGE) != 0)
          throw std::runtime_error("madvise failed");
        std::memset(host_samples[i], 1, registered_bytes);
        report_pages(host_samples[i], registered_bytes, i, "before_registration");
        check(cudaHostRegister(host_samples[i], registered_bytes,
                               cudaHostRegisterDefault));
      } else {
        check(cudaHostAlloc(&host_samples[i], samples + (combined ? scales : 0),
                            host_flags));
      }
      if (combined)
        host_scales[i] = static_cast<char *>(host_samples[i]) + samples;
      else
        check(cudaHostAlloc(&host_scales[i], scales, host_flags));
      if (!registered) std::memset(host_samples[i], 1, samples);
      if (scales) std::memset(host_scales[i], 0, scales);
      check(cudaMalloc(&device_samples[i], samples + (combined ? scales : 0)));
      if (combined)
        device_scales[i] = static_cast<char *>(device_samples[i]) + samples;
      else
        check(cudaMalloc(&device_scales[i], scales));
      check(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
      if (kernel_blocks) check(cudaHostGetDevicePointer(&mapped_device[i], host_samples[i], 0));
      if (dma_percent) check(cudaStreamCreateWithFlags(&kernel_streams[i], cudaStreamNonBlocking));
    }
    cudaPointerAttributes attributes{};
    check(cudaPointerGetAttributes(&attributes, host_samples[0]));
    if (attributes.type != cudaMemoryTypeHost)
      throw std::runtime_error("H2D source is not CUDA-registered host memory");
    unsigned actual_flags = 0;
    check(cudaHostGetFlags(&actual_flags, host_samples[0]));
    std::cout << "cuda_host_pointer_type=host cuda_host_flags=" << actual_flags << '\n';
    if (write_combined && !(actual_flags & cudaHostAllocWriteCombined))
      throw std::runtime_error("write-combined allocation flag missing");
    for (int i = 0; i < buffers; ++i)
      report_pages(host_samples[i], samples + (combined ? scales : 0), i, "pinned");
    auto run = [&] {
      for (int n = 0; n < repeats; ++n) {
        const int i = n % buffers;
        // Keep each buffer on one stream, including when the buffer count
        // is not a multiple of the number of streams.
        cudaStream_t stream = streams[i % stream_count];
        const size_t total = samples + (combined ? scales : 0);
        if (kernel_blocks) {
          const size_t prefix = total * dma_percent / 100 / sizeof(uint4) * sizeof(uint4);
          if (prefix) check(cudaMemcpyAsync(device_samples[i], host_samples[i], prefix,
                                           cudaMemcpyHostToDevice, stream));
          // The two engines touch disjoint ranges, with each range always
          // queued on the same stream. The final device sync joins both.
          copy_mapped_host<<<kernel_blocks, 256, 0, dma_percent ? kernel_streams[i] : stream>>>(
              static_cast<char *>(device_samples[i]) + prefix,
              static_cast<const char *>(mapped_device[i]) + prefix, total - prefix);
          check(cudaGetLastError());
          continue;
        }
        const size_t chunk = chunk_bytes ? std::min(chunk_bytes, total) : total;
        for (size_t offset = 0; offset < total; offset += chunk)
          check(cudaMemcpyAsync(
              static_cast<char *>(d2h ? host_samples[i] : device_samples[i]) + offset,
              static_cast<char *>(d2h ? device_samples[i] : host_samples[i]) + offset,
              std::min(chunk, total - offset),
              d2h ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, stream));
        if (!combined)
          check(cudaMemcpyAsync(d2h ? host_scales[i] : device_scales[i],
                                d2h ? device_scales[i] : host_scales[i], scales,
                                d2h ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice,
                                stream));
      }
      check(cudaDeviceSynchronize());
    };
    run(); // warm up
    const auto timed_start_unix_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const auto begin = std::chrono::steady_clock::now();
    run();
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    const auto timed_end_unix_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (!d2h) {
      if (combined) {
        unsigned *errors = nullptr, host_errors = 0;
        check(cudaMalloc(&errors, sizeof(unsigned)));
        check(cudaMemset(errors, 0, sizeof(unsigned)));
        for (int i = 0; i < buffers && i < repeats; ++i)
          validate_copy<<<256, 256>>>(static_cast<const unsigned char *>(device_samples[i]),
                                     samples, scales, errors);
        check(cudaGetLastError());
        check(cudaMemcpy(&host_errors, errors, sizeof(unsigned), cudaMemcpyDeviceToHost));
        check(cudaFree(errors));
        if (host_errors) throw std::runtime_error("H2D full data check failed");
        std::cout << "h2d_full_data_check=passed\n";
      }
      // Untimed spot checks confirm completed transfers contain the source
      // pattern without reading CPU write-combined memory back on the host.
      for (int i = 0; i < buffers && i < repeats; ++i) {
        for (size_t offset : {size_t(0), samples / 2, samples - 1}) {
          unsigned char value = 0;
          check(cudaMemcpy(&value, static_cast<char *>(device_samples[i]) + offset,
                           1, cudaMemcpyDeviceToHost));
          if (value != 1) throw std::runtime_error("H2D sample spot check failed");
        }
        if (scales) {
          unsigned char value = 1;
          check(cudaMemcpy(&value, static_cast<char *>(device_scales[i]) + scales - 1,
                           1, cudaMemcpyDeviceToHost));
          if (value != 0) throw std::runtime_error("H2D scale spot check failed");
        }
      }
      std::cout << "h2d_data_spot_checks=passed\n";
    }
    std::cout << "direction=" << (d2h ? "d2h" : "h2d")
              << " layout=" << (combined ? "combined" : "split")
              << " channels=" << channels << " streams=" << stream_count
              << " host_memory=" << (no_thp ? "registered_no_thp" :
                    registered ? "registered_thp" :
                    write_combined ? "write_combined" : "default")
              << " samples_bytes=" << samples
              << " scales_bytes=" << scales << " buffers=" << buffers
              << " chunk_bytes=" << chunk_bytes
              << " kernel_blocks=" << kernel_blocks << " rounded=" << rounded
              << " dma_percent=" << dma_percent
              << " repeats=" << repeats << " seconds=" << elapsed
              << " timed_start_unix_ns=" << timed_start_unix_ns
              << " timed_end_unix_ns=" << timed_end_unix_ns
              << " blocks_per_s=" << repeats / elapsed
              << " GB_per_s=" << repeats * double(samples + scales) / elapsed / 1e9
              << std::endl;
    for (int i = 0; i < buffers; ++i) {
      cudaStreamDestroy(streams[i]);
      if (dma_percent) cudaStreamDestroy(kernel_streams[i]);
      cudaFree(device_samples[i]);
      if (!combined) cudaFree(device_scales[i]);
      if (registered) {
        cudaHostUnregister(host_samples[i]);
        munmap(mapped_base[i], registered_bytes + alignment);
      } else {
        cudaFreeHost(host_samples[i]);
      }
      if (!combined) cudaFreeHost(host_scales[i]);
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
