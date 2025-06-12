#include "spatial/spatial.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <cudawrappers/cu.hpp>
#include <libtcc/Correlator.h>
#include <complex>
#include <cuda_fp16.h>

#define NR_BITS  8
#define NR_RECEIVERS 576
#define NR_CHANNELS  480
#define NR_SAMPLES_PER_CHANNEL  3072
#define NR_POLARIZATIONS  2
#define NR_RECEIVERS_PER_BLOCK  64
#define NR_TIMES_PER_BLOCK  128 / NR_BITS
#define NR_BASELINES  NR_RECEIVERS * (NR_RECEIVERS + 1) / 2
int add(int a, int b)
{
    return a + b;
}

template <typename T>
void eigendecomposition(float *h_eigenvalues, int n, const std::vector<T> *A)
{
    T *d_A;
    cudaMalloc((void **)&d_A, n * n * sizeof(T));
    cudaMemcpy(d_A, A->data(), sizeof(T) * n * n, cudaMemcpyHostToDevice);

    //// Allocate memory for eigenvalues and eigenvectors
    float *d_eigenvalues;
    T *d_eigenvectors;
    cudaMalloc((void **)&d_eigenvalues, n * sizeof(float));
    cudaMalloc((void **)&d_eigenvectors, n * n * sizeof(T));

    // float* h_eigenvalues;
    // cudaMallocHost((void**)&h_eigenvalues, n * sizeof(float));

    //
    //// Create cuSOLVER handle
    cusolverDnHandle_t solverHandle;
    cusolverDnCreate(&solverHandle);
    //
    //// Compute eigenvalues and eigenvectors
    int *d_info = nullptr;
    int info = 0;
    cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));

    cudaDataType data_type = CUDA_C_32F;

    cusolverDnParams_t params = NULL;
    cusolverDnCreateParams(&params);

    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;
    void *d_work = nullptr;
    void *h_work = nullptr;

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    cusolverDnXsyevd_bufferSize(
        solverHandle, params, jobz, uplo, n,
        data_type,
        d_A, n, CUDA_R_32F,
        d_eigenvalues, data_type, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);

    cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice);
    h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));

    cusolverDnXsyevd(solverHandle,
                     params,
                     jobz,          // Mode - CUSOLVER_EIG_MODE_VECTOR = get eigenvalues & eigenvectors
                     uplo,          // cublasFillMode_t
                     n,             // size of symmetric matrix
                     data_type,     // data type
                     d_A,           // what to decompose
                     n,             // lda
                     CUDA_R_32F,    // data type output - should always be real for the eigenvalue outputs.
                     d_eigenvalues, // array to store eigenvalues
                     data_type,     // data type of computation
                     d_work, workspaceInBytesOnDevice,
                     h_work, workspaceInBytesOnHost,
                     d_info); //

    //// Check for errors
    int h_info;
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_info != 0)
    {
        std::cerr << "Eigenvalue decomposition failed!" << std::endl;
        return;
    }

    cudaMemcpy(h_eigenvalues, d_eigenvalues, n * sizeof(float), cudaMemcpyDeviceToHost);
    //

    printf("Eigenvalues...\n");
    for (int i = 0; i < n; i++)
    {
        printf("%f \n", h_eigenvalues[i]);
    }

    //// Destroy cuSOLVER handle
    cusolverDnDestroy(solverHandle);
}

template void eigendecomposition<cuComplex>(float *h_eigenvalues, int n, const std::vector<cuComplex> *A);

inline void checkCudaCall(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        std::cerr << "error " << error << std::endl;
        exit(1);
    }
}

void correlate()
{
    // Taken from simpleExample
    typedef std::complex<__half> Sample;
    typedef std::complex<float> Visibility;
    constexpr tcc::Format inputFormat = tcc::Format::fp16;

    typedef Sample Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_RECEIVERS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
    typedef Visibility Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];

    checkCudaCall(cudaSetDevice(0)); // combine the CUDA runtime API and CUDA driver API
    checkCudaCall(cudaFree(0));

    tcc::Correlator correlator(cu::Device(0), inputFormat, NR_RECEIVERS, NR_CHANNELS, NR_SAMPLES_PER_CHANNEL, NR_POLARIZATIONS, NR_RECEIVERS_PER_BLOCK);

    cudaStream_t stream;
    Samples *samples;
    Visibilities *visibilities;

    checkCudaCall(cudaStreamCreate(&stream));
    checkCudaCall(cudaMallocManaged(&samples, sizeof(Samples)));
    checkCudaCall(cudaMallocManaged(&visibilities, sizeof(Visibilities)));

    (*samples)[NR_CHANNELS / 3][NR_SAMPLES_PER_CHANNEL / 5 / NR_TIMES_PER_BLOCK][174][0][NR_SAMPLES_PER_CHANNEL / 5 % NR_TIMES_PER_BLOCK] = Sample(2, 3);
    (*samples)[NR_CHANNELS / 3][NR_SAMPLES_PER_CHANNEL / 5 / NR_TIMES_PER_BLOCK][418][0][NR_SAMPLES_PER_CHANNEL / 5 % NR_TIMES_PER_BLOCK] = Sample(4, 5);

    correlator.launchAsync((CUstream)stream, (CUdeviceptr)visibilities, (CUdeviceptr)samples);
    checkCudaCall(cudaDeviceSynchronize());

    std::cout << ((*visibilities)[160][87745][0][0] == Visibility(23, 2) ? "success" : "failed") << std::endl;

    checkCudaCall(cudaFree(visibilities));
    checkCudaCall(cudaFree(samples));
    checkCudaCall(cudaStreamDestroy(stream));
}