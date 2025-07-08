#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <stdio.h>
#include <complex>
#include <cuda_fp16.h>

int main() {
    const int M = 1024, N = 1024, K = 1024;
    std::complex<__half> h_A[M*K];
    std::complex<__half> h_B[K*N];
    cuComplex h_C[M*N];

for (auto i = 0; i < M*K; ++i) {
    h_A[i] = std::complex<__half>(__float2half(i * 1.0f), __float2half(i * 1.0f));


    }

for (auto i = 0; i < N*K; ++i) {
    h_B[i] = std::complex<__half>(__float2half(i * 1.0f), __float2half(i * 1.0f));

    }


    std::complex<__half> *d_A, *d_B;
    cuComplex  *d_C;
    cudaMalloc(&d_A, sizeof(h_A));
    cudaMalloc(&d_B, sizeof(h_B));
    cudaMalloc(&d_C, sizeof(h_C));
    cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyDefault);
    cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyDefault);
    cudaMemcpy(d_C, h_C, sizeof(h_C), cudaMemcpyDefault);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set math mode for Tensor Core acceleration
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    cuComplex beta  = make_cuComplex(0.0f, 0.0f);

    // cublasCgemmEx for complex single-precision, with fast 16F compute
    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_A, CUDA_C_16F, M,
        d_B, CUDA_C_16F, K,
        &beta,
        d_C, CUDA_C_32F, M,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
    );

    cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost);

    printf("Result:\n");
    for(int i=0; i<M*N; ++i)
        printf("(%f, %f)\n", cuCrealf(h_C[i]), cuCimagf(h_C[i]));

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
