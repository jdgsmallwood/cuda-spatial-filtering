#include "spatial/tensor.hpp"
#include <assert.h>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <vector>

int main(int argc, char **argv) {
  // Host element type definition
  typedef float floatTypeA;
  typedef float floatTypeC;
  typedef float floatTypeCompute;

  // CUDA types
  cutensorDataType_t typeA = CUTENSOR_R_32F;
  cutensorDataType_t typeC = CUTENSOR_R_32F;
  cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

  floatTypeCompute alpha = (floatTypeCompute)1.0f;

  std::vector<int> modeC{'c', 'w', 'h', 'n'};
  std::vector<int> modeA{'w', 'h', 'c', 'n'};

  std::unordered_map<int, int64_t> extent;
  extent['h'] = 1;
  extent['w'] = 4;
  extent['c'] = 4;
  extent['n'] = 1;

  CutensorSetup tensor(extent, CUTENSOR_R_32F, 128);
  tensor.addTensor(modeA, "A");
  tensor.addTensor(modeC, "C");

  auto T_A = tensor.getTensor("A");
  auto T_C = tensor.getTensor("C");

  void *d_A, *d_C;
  cudaMalloc((void **)&d_A, T_A->sizeBytes);
  cudaMalloc((void **)&d_C, T_C->sizeBytes);

  floatTypeA *A, *C;
  cudaMallocHost((void **)&A, sizeof(floatTypeA) * T_A->elements);
  cudaMallocHost((void **)&C, sizeof(floatTypeC) * T_C->elements);

  // initialize data

  for (auto i = 0; i < T_A->elements; i++) {
    A[i] = (((float)rand()) / RAND_MAX) * 100;
  }
  cudaMemcpy2DAsync(d_A, T_A->sizeBytes, A, T_A->sizeBytes, T_A->sizeBytes, 1,
                    cudaMemcpyDefault, nullptr);

  tensor.addPermutation("A", "C", descCompute, "AToC");

  tensor.runPermutation("AToC", alpha, d_A, d_C, nullptr);
  cudaDeviceSynchronize();
  cudaMemcpy(C, d_C, T_C->sizeBytes, cudaMemcpyDefault);
  cudaDeviceSynchronize();

  for (auto i = 0; i < T_A->elements; ++i) {
    std::cout << i << ": " << A[i] << '\n';
  }

  for (auto i = 0; i < T_C->elements; ++i) {
    std::cout << i << ": " << C[i] << '\n';
  }

  cudaFreeHost(A);
  cudaFreeHost(C);
  cudaFree(d_A);
  cudaFree(d_C);

  return 0;
}
