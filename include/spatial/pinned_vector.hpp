#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace cuda_util {

template <typename T> struct CudaPinnedAllocator {
  using value_type = T;

  CudaPinnedAllocator() = default;

  template <typename U>
  constexpr CudaPinnedAllocator(const CudaPinnedAllocator<U> &) noexcept {}

  [[nodiscard]] T *allocate(std::size_t n) {
    if (n == 0)
      return nullptr;

    void *ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, n * sizeof(T), cudaHostAllocDefault);
    if (err != cudaSuccess)
      throw std::runtime_error("cudaHostAlloc failed: " +
                               std::string(cudaGetErrorString(err)));

    return static_cast<T *>(ptr);
  }

  void deallocate(T *ptr, std::size_t) noexcept {
    if (ptr)
      cudaFreeHost(ptr);
  }

  template <typename U>
  bool operator==(const CudaPinnedAllocator<U> &) const noexcept {
    return true;
  }

  template <typename U>
  bool operator!=(const CudaPinnedAllocator<U> &) const noexcept {
    return false;
  }
};

template <typename T>
using PinnedVector = std::vector<T, CudaPinnedAllocator<T>>;

} // namespace cuda_util
