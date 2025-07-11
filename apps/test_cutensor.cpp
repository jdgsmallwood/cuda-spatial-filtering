#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cutensor.h>

#include <unordered_map>
#include <vector>


int main(int argc, char** argv)
{
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
    
    int nmodeA = modeA.size();
    int nmodeC = modeC.size();

    std::unordered_map<int,int64_t> extent;
    extent['h'] = 1;
    extent['w'] = 4;
    extent['c'] = 4;
    extent['n'] = 1;

    std::vector<int64_t> extentA;
    for (auto mode: modeA)
        extentA.push_back(extent[mode]);

    std::vector<int64_t> extentC;
    for (auto mode: modeC)
        extentC.push_back(extent[mode]);

    size_t elementsA = 1;
    for (auto mode: modeA)
         elementsA *= extent[mode];

    size_t elementsC = 1;
    for (auto mode: modeC)
        elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeC = sizeof(floatTypeC) * elementsC;
    
    void *d_A, *d_C;
    cudaMalloc((void**) &d_A, sizeA);
    cudaMalloc((void**) &d_C, sizeC);

    uint32_t const kAlignment = 128;


    floatTypeA *A, *C;
    cudaMallocHost((void**) &A, sizeof(floatTypeA) * elementsA);
    cudaMallocHost((void**) &C, sizeof(floatTypeC) * elementsC);


    // initialize data
    
    for (auto i =0; i < elementsA; i++) {
        A[i] = (((float) rand()) / RAND_MAX) * 100;
    }
    cudaMemcpy2DAsync(d_A, sizeA, A, sizeA, sizeA, 1, cudaMemcpyDefault, nullptr);


    cutensorHandle_t handle;

    cutensorCreate(&handle);

    cutensorTensorDescriptor_t descA;
    cutensorCreateTensorDescriptor(handle, &descA, nmodeA, extentA.data(), nullptr, typeA, kAlignment);

    cutensorTensorDescriptor_t descC;
    cutensorCreateTensorDescriptor(handle, &descC, nmodeC, extentC.data(), nullptr, typeC, kAlignment);

    cutensorOperationDescriptor_t desc;
    cutensorCreatePermutation(handle, &desc, descA, modeA.data(), CUTENSOR_OP_IDENTITY, descC, modeC.data(), descCompute);

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t planPref;
    cutensorCreatePlanPreference(handle, &planPref, algo, CUTENSOR_JIT_MODE_NONE);

    cutensorPlan_t plan;
    cutensorCreatePlan(handle, &plan, desc, planPref, 0);
    
    cutensorPermute(handle, plan, &alpha, d_A, d_C, nullptr);
  
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDefault);
    cudaDeviceSynchronize();

    for (auto i = 0; i < elementsA; ++i) {
      std::cout << i << ": " << A[i] << '\n';
  }

    for (auto i = 0; i < elementsC; ++i) {
      std::cout << i << ": " << C[i] << '\n';
  }
    

    cutensorDestroy(handle);
    cutensorDestroyPlan(plan);
    cutensorDestroyOperationDescriptor(desc);
    cutensorDestroyPlanPreference(planPref);
    cutensorDestroyTensorDescriptor(descA);
    cutensorDestroyTensorDescriptor(descC);

    cudaFreeHost(A);
    cudaFreeHost(C);
    cudaFree(d_A);
    cudaFree(d_C);



    return 0;
}
