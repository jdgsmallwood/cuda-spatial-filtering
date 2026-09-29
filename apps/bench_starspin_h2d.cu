// Starspin's actual samples + scales transfer sizes, with pinned host buffers.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <vector>

static void check(cudaError_t err) {
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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
    if ((!raw_bytes && (channels < 1 || channels > 60)) ||
        (raw_bytes && (requested_bytes < 1 || requested_bytes > 14000000000ULL)) ||
        repeats < 1 || repeats > 1000 || (stream_count != 1 && stream_count != 3) ||
        buffers < 1 || buffers > 3 || stream_count > buffers ||
        (raw_bytes && !combined) || (registered && !combined))
      throw std::runtime_error("usage: bench_starspin_h2d [channels 1..60|bytes:N<=14000000000] [repeats 1..1000] [split|combined] [1|3 streams] [default|wc|register|register_no_thp] [buffers 1..3] [h2d|d2h]");
    constexpr size_t packets = 256 + 2, fpgas = 4, receivers = 40, pols = 2;
    constexpr size_t samples_per_packet = 64;
    const size_t samples = raw_bytes ? requested_bytes :
        size_t(channels) * packets * fpgas * samples_per_packet *
        (receivers / fpgas) * pols * 2;
    const size_t scales = raw_bytes ? 0 :
        size_t(channels) * packets * receivers * pols * 2;
    const unsigned host_flags = write_combined ? cudaHostAllocWriteCombined
                                                : cudaHostAllocDefault;
    std::vector<void *> host_samples(buffers), host_scales(buffers);
    std::vector<void *> device_samples(buffers), device_scales(buffers);
    std::vector<void *> mapped_base(buffers, nullptr);
    std::vector<cudaStream_t> streams(buffers);
    for (int i = 0; i < buffers; ++i) {
      if (registered) {
        constexpr size_t alignment = 2 * 1024 * 1024;
        const size_t alloc_bytes = samples + scales + alignment;
        mapped_base[i] = mmap(nullptr, alloc_bytes, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped_base[i] == MAP_FAILED) throw std::runtime_error("mmap failed");
        host_samples[i] = reinterpret_cast<void *>(
            (reinterpret_cast<uintptr_t>(mapped_base[i]) + alignment - 1) &
            ~(uintptr_t(alignment) - 1));
        madvise(host_samples[i], samples + scales,
                no_thp ? MADV_NOHUGEPAGE : MADV_HUGEPAGE);
        std::memset(host_samples[i], 1, samples + scales);
        check(cudaHostRegister(host_samples[i], samples + scales,
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
    }
    cudaPointerAttributes attributes{};
    check(cudaPointerGetAttributes(&attributes, host_samples[0]));
    if (attributes.type != cudaMemoryTypeHost)
      throw std::runtime_error("H2D source is not CUDA-registered host memory");
    auto run = [&] {
      for (int n = 0; n < repeats; ++n) {
        const int i = n % buffers;
        cudaStream_t stream = streams[n % stream_count];
        check(cudaMemcpyAsync(d2h ? host_samples[i] : device_samples[i],
                              d2h ? device_samples[i] : host_samples[i],
                              samples + (combined ? scales : 0),
                              d2h ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice,
                              stream));
        if (!combined)
          check(cudaMemcpyAsync(d2h ? host_scales[i] : device_scales[i],
                                d2h ? device_scales[i] : host_scales[i], scales,
                                d2h ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice,
                                stream));
      }
      check(cudaDeviceSynchronize());
    };
    run(); // warm up
    const auto begin = std::chrono::steady_clock::now();
    run();
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    std::cout << "direction=" << (d2h ? "d2h" : "h2d")
              << " layout=" << (combined ? "combined" : "split")
              << " channels=" << channels << " streams=" << stream_count
              << " host_memory=" << (no_thp ? "registered_no_thp" :
                    registered ? "registered_thp" :
                    write_combined ? "write_combined" : "default")
              << " samples_bytes=" << samples
              << " scales_bytes=" << scales << " buffers=" << buffers
              << " repeats=" << repeats << " seconds=" << elapsed
              << " blocks_per_s=" << repeats / elapsed
              << " GB_per_s=" << repeats * double(samples + scales) / elapsed / 1e9
              << std::endl;
    for (int i = 0; i < buffers; ++i) {
      cudaStreamDestroy(streams[i]);
      cudaFree(device_samples[i]);
      if (!combined) cudaFree(device_scales[i]);
      if (registered) {
        cudaHostUnregister(host_samples[i]);
        munmap(mapped_base[i], samples + scales + 2 * 1024 * 1024);
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
