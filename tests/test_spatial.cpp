#include <gtest/gtest.h>
#include "spatial/spatial.hpp"
#include <vector>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <ccglib/ccglib.hpp>
#include "ccglib/precision.h"

#include <cudawrappers/cu.hpp>
TEST(EigenvalueDecompositionTest, SimpleValueTest)
{

    const std::vector<cuComplex> A = {
        make_cuComplex(3.5, 0.0), make_cuComplex(0.5, 1.0), make_cuComplex(0.0, 0.0),
        make_cuComplex(0.5, -1.0), make_cuComplex(3.5, 0.0), make_cuComplex(0.0, 0.0),
        make_cuComplex(0.0, 0.0), make_cuComplex(0.0, 0.0), make_cuComplex(2.0, 0)};

    float *h_eigenvalues;
    cudaMallocHost((void **)&h_eigenvalues, 3 * sizeof(float));

    eigendecomposition<cuComplex>(h_eigenvalues, 3, &A);

    EXPECT_FLOAT_EQ(h_eigenvalues[0], 2.0);
    EXPECT_FLOAT_EQ(h_eigenvalues[1], 2.381966);
    EXPECT_FLOAT_EQ(h_eigenvalues[2], 4.618034);
}

TEST(CorrelateTest, SimpleTest)
{
    Samples *samples;

    try {
    std::cout << "Creating samples..." << std::endl;
    checkCudaCall(cudaMallocHost(&samples, sizeof(Samples)));
    (*samples)[1][0][4][0][0] = Sample(2, 3);
    (*samples)[1][0][5][0][0] = Sample(4, 5);
    std::cout << "Creating visibilities" << std::endl;
    Visibilities *visibilities;
    checkCudaCall(cudaMallocHost(&visibilities, sizeof(Visibilities)));
    std::cout << "Starting correlation..." << std::endl; 
    correlate(samples, visibilities);
    std::cout << "Finished correlation..." << std::endl;
    
    print_nonzero_visibilities(visibilities);
    
    EXPECT_EQ((*visibilities)[1][19][0][0], Visibility(23, 2));
    EXPECT_EQ((*visibilities)[1][14][0][0], Visibility(13, 0));
    EXPECT_EQ((*visibilities)[1][20][0][0], Visibility(41, 0));

    checkCudaCall(cudaFreeHost(visibilities));

    checkCudaCall(cudaFreeHost(samples));
    } catch (std::exception &error) {
        std::cerr << error.what() << std::endl;
    }
}


TEST(CorrelateTest, SimpleTestExtended)
{
    Samples *samples;

    try {
    std::cout << "Creating samples..." << std::endl;
    checkCudaCall(cudaMallocHost(&samples, sizeof(Samples)));
    
    
    (*samples)[1][0][4][0][0] = Sample(2, 3);
    (*samples)[1][0][4][0][1] = Sample(5, 1);
    (*samples)[1][0][4][0][2] = Sample(-2, 3);
    (*samples)[1][0][4][0][3] = Sample(2, -3);
    (*samples)[1][0][4][0][4] = Sample(1, 1);
    (*samples)[1][0][4][0][5] = Sample(0, 0);
    (*samples)[1][0][4][0][6] = Sample(10, 2);
    (*samples)[1][0][4][0][7] = Sample(4, 4);
    
    (*samples)[1][0][5][0][0] = Sample(4, 5);
    (*samples)[1][0][5][0][1] = Sample(2, 2);
    (*samples)[1][0][5][0][2] = Sample(9, 1);
    (*samples)[1][0][5][0][3] = Sample(-4, -5);
    (*samples)[1][0][5][0][4] = Sample(0, 0);
    (*samples)[1][0][5][0][5] = Sample(1, -1);
    (*samples)[1][0][5][0][6] = Sample(2, 2);
    (*samples)[1][0][5][0][7] = Sample(2, 2);
    std::cout << "Creating visibilities" << std::endl;
    Visibilities *visibilities;
    checkCudaCall(cudaMallocHost(&visibilities, sizeof(Visibilities)));
    std::cout << "Starting correlation..." << std::endl; 
    correlate(samples, visibilities);
    std::cout << "Finished correlation..." << std::endl;
    
    print_nonzero_visibilities(visibilities);
    // Python code to verify result: 
    // >>> import numpy as np
    // >>> i = np.array([2 + 3j, 5 + 1j, -2 + 3j, 2 -3j, 1 + 1j, 0, 10 + 2j, 4 + 4j])
    // >>> i2 = np.array([4 + 5j, 2 + 2j, 9 + 1j, -4 -5j, 0, 1 -1j, 2 + 2j, 2+ 2j])
    // >>> np.dot(i, i2.conj())
    //  (67+29j)
    // >>> np.dot(i, i.conj())
    //  (203+0j)
    // >>> np.dot(i2, i2.conj())
    //  (190+0j)

    EXPECT_EQ((*visibilities)[1][19][0][0], Visibility(67, 29));
    EXPECT_EQ((*visibilities)[1][14][0][0], Visibility(203, 0));
    EXPECT_EQ((*visibilities)[1][20][0][0], Visibility(190, 0));

    checkCudaCall(cudaFreeHost(visibilities));

    checkCudaCall(cudaFreeHost(samples));
    } catch (std::exception &error) {
        std::cerr << error.what() << std::endl;
    }


}



TEST(CCGLIBTest, SimpleTest)
{   
    cu::init();
    cu::Device device(0);
    cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
    constexpr int n_row = 2;
    constexpr int n_col = 2;   
    constexpr int batch_size = 1; 
    __half (*A);

    checkCudaCall(cudaMallocHost(&A, sizeof(__half) * 2 *  n_row * n_col * batch_size));

    __half (*B);
    
    checkCudaCall(cudaMallocHost(&B, sizeof(__half) * 2  * n_row * n_col * batch_size));

    for (auto i = 0; i <batch_size * 2 * n_row * n_col; i++) {
        A[i] = 1;
        B[i] = 1;
    }
    
    float (*C);
    
    checkCudaCall(cudaMallocHost(&C, sizeof(float) * 2 * n_row * n_col * batch_size));

    __half (*d_A);
    checkCudaCall(cudaMalloc(&d_A, sizeof(__half) * 2 *n_row*n_col*batch_size));

    __half (*d_B);
    checkCudaCall(cudaMalloc(&d_B, sizeof(__half) * 2*n_row*n_col*batch_size));

    float (*d_C);

    checkCudaCall(cudaMalloc(&d_C, sizeof(float) * 2*n_row*n_col*batch_size));

    checkCudaCall(cudaSetDevice(0)); // combine the CUDA runtime API and CUDA driver API
    cudaStream_t stream;
    checkCudaCall(cudaStreamCreate(&stream));
    checkCudaCall(cudaMemcpyAsync(d_A, A, sizeof(__half) * 2  * n_row * n_col * batch_size , cudaMemcpyHostToDevice, stream));
    checkCudaCall(cudaMemcpyAsync(d_B, B, sizeof(__half) * 2  * n_row * n_col * batch_size , cudaMemcpyHostToDevice, stream));

    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);

    ccglib::mma::GEMM gemm_mma(batch_size, n_row, n_col, n_row, cu_device,
                             stream, ccglib::ValueType::float16,
                             ccglib::mma::basic);
    
    gemm_mma.Run((CUdeviceptr) d_A, (CUdeviceptr) d_B,(CUdeviceptr) d_C);

    checkCudaCall(cudaMemcpyAsync(C, d_C, sizeof(float) * 2 * n_row * n_col * batch_size, cudaMemcpyDeviceToHost, stream ));

    checkCudaCall(cudaStreamSynchronize(stream));
    
    // for debugging only.
    for (auto i =0; i < batch_size * 2 * n_row * n_col; i++) {
        std::cout << i << "i" << C[i] << std::endl;
    }

    EXPECT_EQ(C[0], 0);
    EXPECT_EQ(C[1], 0);
    EXPECT_EQ(C[2], 0);
    EXPECT_EQ(C[3], 0);
    EXPECT_EQ(C[4], 4);
    EXPECT_EQ(C[5], 4);
    EXPECT_EQ(C[6], 4);
    EXPECT_EQ(C[7], 4);
}