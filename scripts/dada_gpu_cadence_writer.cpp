// Diagnostic: write a PSRDADA GPU ring at the fine-beam block cadence.
// Usage: dada_gpu_cadence_writer KEY_HEX HEADER DSPSR_LOG DURATION_SECONDS
#include <cuda_runtime.h>
extern "C" {
#include "dada_hdu.h"
#include "ipcbuf.h"
#include "ipcio.h"
#include "multilog.h"
}

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void check(cudaError_t status, const char *what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

int main(int argc, char **argv) {
  try {
    if (argc != 5)
      throw std::runtime_error("usage: writer KEY_HEX HEADER DSPSR_LOG SECONDS");
    const key_t key = static_cast<key_t>(std::stoul(argv[1], nullptr, 16));
    const double seconds = std::stod(argv[4]);
    auto *log = multilog_open("dada_gpu_cadence_writer", 0);
    multilog_add(log, stderr);
    auto *hdu = dada_hdu_create(log);
    dada_hdu_set_key(hdu, key);
    if (dada_hdu_connect(hdu) < 0 || dada_hdu_lock_write(hdu) < 0)
      throw std::runtime_error("could not connect and lock DADA ring");
    const uint64_t header_bytes = ipcbuf_get_bufsz(hdu->header_block);
    std::vector<char> header(header_bytes);
    std::ifstream input(argv[2], std::ios::binary);
    if (!input.read(header.data(), static_cast<std::streamsize>(header_bytes)))
      throw std::runtime_error("could not read complete DADA header");
    char *header_block = ipcbuf_get_next_write(hdu->header_block);
    if (!header_block)
      throw std::runtime_error("could not acquire DADA header block");
    std::memcpy(header_block, header.data(), header_bytes);
    if (ipcbuf_mark_filled(hdu->header_block, header_bytes) < 0)
      throw std::runtime_error("could not publish DADA header");
    std::cout << "header published; waiting for DSPSR preparation" << std::endl;

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(90);
    bool ready = false;
    while (std::chrono::steady_clock::now() < deadline) {
      std::ifstream dspsr_log(argv[3]);
      std::string line;
      while (std::getline(dspsr_log, line))
        if (line.find("prepared in ") != std::string::npos)
          ready = true;
      if (ready) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!ready)
      throw std::runtime_error("DSPSR did not prepare within 90 seconds");

    const uint64_t block_bytes = ipcbuf_get_bufsz((ipcbuf_t *)hdu->data_block);
    // One 512-sample fine-beam block covers 512 * 34.56 microseconds.
    constexpr double block_seconds = 512 * 34.56e-6;
    const size_t total_blocks = static_cast<size_t>(std::ceil(seconds / block_seconds));
    std::vector<float> sample(block_bytes / sizeof(float));
    std::mt19937 rng(314159);
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    for (float &x : sample) x = gaussian(rng);
    void *device_sample = nullptr;
    check(cudaMalloc(&device_sample, block_bytes), "cudaMalloc");
    check(cudaMemcpy(device_sample, sample.data(), block_bytes, cudaMemcpyHostToDevice),
          "initial cudaMemcpy");
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    const auto started = std::chrono::steady_clock::now();
    std::cout << "writing " << total_blocks << " blocks of " << block_bytes
              << " bytes at " << 1.0 / block_seconds << " blocks/s" << std::endl;
    for (size_t i = 0; i < total_blocks; ++i) {
      uint64_t block_id = 0;
      char *block = ipcio_open_block_write(hdu->data_block, &block_id);
      if (!block) throw std::runtime_error("ipcio_open_block_write failed");
      check(cudaMemcpyAsync(block, device_sample, block_bytes,
                            cudaMemcpyDeviceToDevice, stream), "DADA cudaMemcpyAsync");
      check(cudaStreamSynchronize(stream), "DADA cudaStreamSynchronize");
      if (ipcio_close_block_write(hdu->data_block, block_bytes) < 0)
        throw std::runtime_error("ipcio_close_block_write failed");
      if ((i + 1) % 100 == 0)
        std::cout << "wrote " << i + 1 << " blocks" << std::endl;
      std::this_thread::sleep_until(
          started + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>((i + 1) * block_seconds)));
    }
    if (ipcio_stop(hdu->data_block) < 0)
      throw std::runtime_error("ipcio_stop failed");
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << "finished " << total_blocks << " blocks in " << elapsed
              << " seconds" << std::endl;
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    check(cudaFree(device_sample), "cudaFree");
    dada_hdu_unlock_write(hdu);
    dada_hdu_disconnect(hdu);
    dada_hdu_destroy(hdu);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "DADA GPU writer failed: " << error.what() << std::endl;
    return 1;
  }
}
