#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

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

    printf("Include headers and define data types\n");


    std::vector<int> modeC{'c', 'w', 'h', 'n'};
    std::vector<int> modeA{'w', 'h', 'c', 'n'};
    
    int nmodeA = modeA.size();
    int nmodeC = modeC.size();

    std::unordered_map<int,int64_t> extent;
    extent['h'] = 128;
    extent['w'] = 32;
    extent['c'] = 128;
    extent['n'] = 128;

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

    return 0;
}
