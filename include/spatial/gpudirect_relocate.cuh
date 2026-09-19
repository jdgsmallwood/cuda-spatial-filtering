#pragma once

#include <cstdint>

// Device-side relocation copy shared by LibibverbsGpuDirectPacketCapture
// (libibverbs.hpp, HAVE_IBVERBS-gated) and its correctness test
// (tests/test_gpudirect_scatter.cu, which doesn't need real ibverbs
// hardware/headers to exercise this kernel). See
// /home/ubuntu/.claude/plans/i-want-to-start-breezy-lampson.md.
//
// One thread block per packet, plain byte loop -- payload sizes (e.g. 40B
// scales) aren't guaranteed 16-byte aligned, so this doesn't assume float4
// vectorization is safe the way the Phase 0 benchmark's fixed-size slots
// could. `src`/`dst` are parallel arrays of per-packet device pointers,
// themselves read through a cudaHostAllocMapped mirror so the host can
// update them each batch with no explicit H2D copy.
// static (internal linkage): this header may be included by more than one
// .cu translation unit; without static, a non-template __global__ function
// defined in a header risks duplicate-symbol link errors if two such TUs
// ever end up in the same binary.
static __global__ void gpudirect_relocate_kernel(void *const *__restrict__ src,
                                                 void *const *__restrict__ dst,
                                                 int slot_bytes) {
  const uint8_t *s = static_cast<const uint8_t *>(src[blockIdx.x]);
  uint8_t *d = static_cast<uint8_t *>(dst[blockIdx.x]);
  for (int i = threadIdx.x; i < slot_bytes; i += blockDim.x) {
    d[i] = s[i];
  }
}
