#include "spatial/spatial.hpp"
#include <cuda_runtime.h>


int add(int a, int b) {
    return a + b;
}


void eigendecomposition<template T>(float* h_eigenvalues) {
T* d_A;
cudaMalloc((void**) &d_A, n*n*sizeof(T));
cudaMemcpy(d_A, A.data(), sizeof(T) * n * n, cudaMemcpyHostToDevice);


//// Allocate memory for eigenvalues and eigenvectors
float* d_eigenvalues;
u_data_type* d_eigenvectors;
cudaMalloc((void**)&d_eigenvalues, n * sizeof(float));
cudaMalloc((void**)&d_eigenvectors, n * n * sizeof(T));

//float* h_eigenvalues;
//cudaMallocHost((void**)&h_eigenvalues, n * sizeof(float));

//
//// Create cuSOLVER handle
cusolverDnHandle_t solverHandle;
cusolverDnCreate(&solverHandle);
//
//// Compute eigenvalues and eigenvectors
int* d_info = nullptr;
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
    d_eigenvalues, data_type, &workspaceInBytesOnDevice, &workspaceInBytesOnHost
);

cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice);
h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));


cusolverDnXsyevd(solverHandle,
    params, 
    jobz, // Mode - CUSOLVER_EIG_MODE_VECTOR = get eigenvalues & eigenvectors
    uplo, // cublasFillMode_t 
    n, // size of symmetric matrix
    data_type, // data type
    d_A, // what to decompose
    n, //lda
    CUDA_R_32F, // data type output - should always be real for the eigenvalue outputs.
    d_eigenvalues, // array to store eigenvalues
    data_type, // data type of computation
    d_work, workspaceInBytesOnDevice,
    h_work, workspaceInBytesOnHost,
    d_info
);    //



//// Check for errors
int h_info;
cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
if (h_info != 0) {
    std::cerr << "Eigenvalue decomposition failed!" << std::endl;
    return -1;
}

cudaMemcpy(h_eigenvalues, d_eigenvalues, n * sizeof(float), cudaMemcpyDeviceToHost);
//


printf("Eigenvalues...\n");
for (int i = 0; i < n; i++) {
    printf("%f \n", h_eigenvalues[i]);
}

//// Destroy cuSOLVER handle
cusolverDnDestroy(solverHandle);
return 0;
}
