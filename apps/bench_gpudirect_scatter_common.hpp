// Shared relocation-stream machinery for the Phase 0 GPUDirect ingest
// validation benchmark (see /home/ubuntu/.claude/plans/i-want-to-start-breezy-lampson.md).
// Used standalone (synthetic busy kernel) by bench_gpudirect_scatter.cu, and
// alongside the real LambdaGPUPipeline/LambdaCorrBeamOnlyGPUPipeline workload
// by bench_gpu.cu, so both a quick synthetic check and a check against the
// actual correlator/beamformer kernels are available.
#pragma once
#include "spatial/logging.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

namespace gpudirect_bench {

// Simplification (deliberate, not an oversight): the landing pool and
// destination arrays are flat byte buffers sized to match one packet's
// samples+scales footprint, not byte-exact InputPacketSamplesType/
// PacketScalesType layouts -- this measures relocation kernel/graph timing
// and memory-traffic volume, not end-to-end placement correctness (that's
// covered separately by tests/test_gpudirect_scatter.cu once the real
// ibverbs integration exists).
constexpr size_t PACKET_SAMPLES_BYTES = 2560; // NR_TIME_STEPS_PER_PACKET * NR_RECEIVERS_PER_PACKET
                                               // * NR_POLARIZATIONS * sizeof(complex<int8_t>), default shapes
constexpr size_t PACKET_SCALES_BYTES = 40;    // NR_RECEIVERS_PER_PACKET * NR_POLARIZATIONS * sizeof(int16_t)
// Padded up to a float4 (16B) multiple so every slot offset (slot_index *
// PACKET_BYTES) stays 16-byte aligned for the scatter kernel's float4 copies
// -- 2560+40=2600 is not itself a multiple of 16.
constexpr size_t PACKET_BYTES = ((PACKET_SAMPLES_BYTES + PACKET_SCALES_BYTES + 15) / 16) * 16;

struct RelocationDescriptor {
  uint32_t src_slot;
  uint32_t dst_slot;
};

// One thread block per packet; threads cooperatively copy PACKET_BYTES via
// float4-width stores. Mirrors the real design's scatter kernel shape: reads
// a descriptor array (host-written, mapped pinned memory -- no per-batch
// memcpy call needed) and relocates each packet's payload from the landing
// pool to its destination slot.
__global__ inline void scatter_kernel(const uint8_t *__restrict__ landing_pool,
                                      uint8_t *__restrict__ destination,
                                      const RelocationDescriptor *__restrict__ descriptors) {
  const RelocationDescriptor d = descriptors[blockIdx.x];
  const float4 *src = reinterpret_cast<const float4 *>(landing_pool + (size_t)d.src_slot * PACKET_BYTES);
  float4 *dst = reinterpret_cast<float4 *>(destination + (size_t)d.dst_slot * PACKET_BYTES);
  constexpr int NR_FLOAT4 = PACKET_BYTES / sizeof(float4); // PACKET_BYTES is a multiple of 16
  for (int i = threadIdx.x; i < NR_FLOAT4; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// Mirrors LambdaGPUPipeline::capture_graph (pipeline/lambda_gpu_pipeline.hpp)
// -- raw CUDA calls (not CUDA_CHECK) so a capture-incompatible op degrades to
// a clear failure instead of aborting the whole benchmark.
template <typename EnqueueFn>
inline bool capture_graph(cudaStream_t stream, EnqueueFn &&enqueue, cudaGraphExec_t &exec_out) {
  if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  enqueue();
  cudaGraph_t graph = nullptr;
  const cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
  if (end_err != cudaSuccess || graph == nullptr) {
    if (graph != nullptr) cudaGraphDestroy(graph);
    cudaGetLastError();
    return false;
  }
  cudaGraphExec_t exec = nullptr;
  if (cudaGraphInstantiateWithFlags(&exec, graph, 0) != cudaSuccess || exec == nullptr) {
    cudaGraphDestroy(graph);
    cudaGetLastError();
    return false;
  }
  cudaGraphDestroy(graph);
  exec_out = exec;
  return true;
}

struct StreamStats {
  unsigned long long replays = 0;
  unsigned long long stalls = 0; // double-buffer half not ready when needed
  double min_replay_us = 1e18;
  double max_replay_us = 0.0;
  double total_replay_us = 0.0;
};

// One simulated FPGA capture thread: owns a double-buffered landing pool
// (2 x poll_batch slots), a destination ring (dest_slots slots), a mapped
// host-pinned descriptor buffer the scatter kernel reads directly (no
// per-batch memcpy call), and a low-priority stream + captured graph.
// Runs flat-out (not throttled to the target cadence) for duration_s to find
// the true achievable replay rate, gated by the real double-buffer safety
// check (cudaEventQuery on the half's previous graph replay).
inline void run_stream(int poll_batch, int dest_slots, double duration_s,
                       std::atomic<bool> &start_gate, StreamStats &stats) {
  int low_prio = 0, high_prio = 0;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, low_prio));

  uint8_t *landing_pool; // 2 halves x poll_batch slots
  uint8_t *destination;  // dest_slots slots
  CUDA_CHECK(cudaMalloc(&landing_pool, 2ull * poll_batch * PACKET_BYTES));
  CUDA_CHECK(cudaMalloc(&destination, (size_t)dest_slots * PACKET_BYTES));

  RelocationDescriptor *descriptors_host = nullptr;
  CUDA_CHECK(cudaHostAlloc((void **)&descriptors_host,
                           sizeof(RelocationDescriptor) * poll_batch,
                           cudaHostAllocMapped));
  RelocationDescriptor *descriptors_dev = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer((void **)&descriptors_dev, descriptors_host, 0));

  cudaGraphExec_t exec = nullptr;
  const bool captured = capture_graph(stream, [&]() {
    scatter_kernel<<<poll_batch, 256, 0, stream>>>(landing_pool, destination, descriptors_dev);
  }, exec);
  if (!captured) {
    fprintf(stderr, "[stream] failed to capture relocation graph\n");
    return;
  }

  cudaEvent_t half_done[2];
  bool half_used[2] = {false, false};
  CUDA_CHECK(cudaEventCreateWithFlags(&half_done[0], cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&half_done[1], cudaEventDisableTiming));

  // Warm up (first replay includes any lazy JIT/allocation cost).
  CUDA_CHECK(cudaGraphLaunch(exec, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  while (!start_gate.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  uint32_t next_dest_slot = 0;
  int half = 0;
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < duration_s) {
    const auto replay_start = std::chrono::steady_clock::now();

    if (half_used[half]) {
      if (cudaEventQuery(half_done[half]) != cudaSuccess) {
        stats.stalls++;
        // Spin-poll, not cudaEventSynchronize: the blocking-wait API has real
        // OS wake-latency (often tens of us) that would swamp this
        // measurement with driver/scheduler noise rather than actual GPU
        // completion time. A real capture thread would busy-poll its
        // completion queue anyway (see libibverbs.hpp's existing
        // ibv_poll_cq loop), so spin-polling here matches the real design.
        while (cudaEventQuery(half_done[half]) != cudaSuccess) {
          // busy-wait
        }
      }
    }

    const uint32_t base_src = half * poll_batch;
    for (int i = 0; i < poll_batch; ++i) {
      descriptors_host[i].src_slot = base_src + i;
      descriptors_host[i].dst_slot = next_dest_slot;
      next_dest_slot = (next_dest_slot + 1) % dest_slots;
    }

    CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaEventRecord(half_done[half], stream));
    half_used[half] = true;

    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - replay_start).count();
    stats.replays++;
    stats.total_replay_us += us;
    stats.min_replay_us = std::min(stats.min_replay_us, us);
    stats.max_replay_us = std::max(stats.max_replay_us, us);

    half ^= 1;
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraphExecDestroy(exec);
  cudaEventDestroy(half_done[0]);
  cudaEventDestroy(half_done[1]);
  cudaFreeHost(descriptors_host);
  cudaFree(landing_pool);
  cudaFree(destination);
  cudaStreamDestroy(stream);
}

} // namespace gpudirect_bench
