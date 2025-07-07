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
    // Testing parameters are:
    // NR_CHANNELS = 3
    // NR_SAMPLES_PER_CHANNEL = 8
    // NR_RECEIVERS = 8
    // NR_POLARIZATIONS = 2
    // NR_RECEIVERS_PER_BLOCK = 64

    // Baselines will be
    // 0-0 (1)
    // 1-0 1-1 (2 + 1 =3 total)
    // 2-0 2-1 2-2  (3 + 3 = 6 total)
    // 3-0 3-1 3-2 3-3 (6 + 4 = 10 total)
    // 4-0 4-1 4-2 4-3 4-4 (10 + 5 = 15 total)
    // 5-0 5-1 5-2 5-3 5-4 5-5 (15 + 6 = 21 total)

    // From above and using zero-indexing then the 4-4 term will be 14
    // the 5-4 term will be 19 and 5-5 term will be 20 as seen below.
    // So the output structure is lower triangular matrix (assuming row-major)
    Samples *samples;

    try
    {
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
    }
    catch (std::exception &error)
    {
        std::cerr << error.what() << std::endl;
    }
}

TEST(CorrelateTest, SimpleTestExtended)
{
    Samples *samples;

    try
    {
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
    }
    catch (std::exception &error)
    {
        std::cerr << error.what() << std::endl;
    }
}

TEST(CCGLIBTest, SimpleTest)
{
    constexpr int n_row = 2;
    constexpr int n_col = 2;
    constexpr int batch_size = 1;
    __half(*A);

    checkCudaCall(cudaMallocHost(&A, sizeof(__half) * 2 * n_row * n_col * batch_size));

    __half(*B);

    checkCudaCall(cudaMallocHost(&B, sizeof(__half) * 2 * n_row * n_col * batch_size));

    for (auto i = 0; i < batch_size * 2 * n_row * n_col; i++)
    {
        A[i] = 1;
        B[i] = 1;
    }

    float(*C);

    checkCudaCall(cudaMallocHost(&C, sizeof(float) * 2 * n_row * n_col * batch_size));

    ccglib_mma(A, B, C, n_row, n_col, batch_size);
    // for debugging only.
    for (auto i = 0; i < batch_size * 2 * n_row * n_col; i++)
    {
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
    checkCudaCall(cudaFreeHost(A));
    checkCudaCall(cudaFreeHost(B));
    checkCudaCall(cudaFreeHost(C));
}
TEST(CCGLIBTest, SimpleTest2)
{
    constexpr int n_row = 2;
    constexpr int n_col = 2;
    constexpr int batch_size = 1;
    __half(*A);

    checkCudaCall(cudaMallocHost(&A, sizeof(__half) * 2 * n_row * n_col * batch_size));

    __half(*B);

    checkCudaCall(cudaMallocHost(&B, sizeof(__half) * 2 * n_row * n_col * batch_size));

    for (auto i = 0; i < batch_size * 2 * n_row * n_col; i++)
    {
        if (i < batch_size * n_row * n_col)
        {
            A[i] = 1;
            B[i] = 1;
        }
        else
        {
            A[i] = 0;
            B[i] = 0;
        }
    }

    float(*C);

    checkCudaCall(cudaMallocHost(&C, sizeof(float) * 2 * n_row * n_col * batch_size));

    ccglib_mma(A, B, C, n_row, n_col, batch_size);

    EXPECT_EQ(C[0], 2);
    EXPECT_EQ(C[1], 2);
    EXPECT_EQ(C[2], 2);
    EXPECT_EQ(C[3], 2);
    EXPECT_EQ(C[4], 0);
    EXPECT_EQ(C[5], 0);
    EXPECT_EQ(C[6], 0);
    EXPECT_EQ(C[7], 0);
    checkCudaCall(cudaFreeHost(A));
    checkCudaCall(cudaFreeHost(B));
    checkCudaCall(cudaFreeHost(C));
}
TEST(CCGLIBTest, SimpleTestMatrixStructure)
{
    constexpr int n_row = 2;
    constexpr int n_col = 2;
    constexpr int batch_size = 1;
    __half(*A);

    checkCudaCall(cudaMallocHost(&A, sizeof(__half) * 2 * n_row * n_col * batch_size));

    __half(*B);

    checkCudaCall(cudaMallocHost(&B, sizeof(__half) * 2 * n_row * n_col * batch_size));

    const int div = n_row * n_col;
    A[0] = 1;
    A[div + 0] = 2;
    A[1] = 2;
    A[div + 1] = 1;
    A[2] = -1;
    A[div + 2] = -2;
    A[3] = 0;
    A[div + 3] = 1;

    B[0] = 1;
    B[div + 0] = 1;
    B[1] = 2;
    B[div + 1] = 4;
    B[2] = -2;
    B[div + 2] = -1;
    B[3] = 2;
    B[div + 3] = 0;

    float(*C);

    checkCudaCall(cudaMallocHost(&C, sizeof(float) * 2 * n_row * n_col * batch_size));

    ccglib_mma(A, B, C, n_row, n_col, batch_size);
    // for debugging only.
    for (auto i = 0; i < batch_size * 2 * n_row * n_col; i++)
    {
        std::cout << i << "i" << C[i] << std::endl;
    }
    // The first matrix "A" is row-major. The second matrix "B" is column-major so
    // we end up doing a @b.T.
    // >>> import numpy as np
    // >>> a = np.array([[1 + 2j, 2 + 1j],[-1 - 2j, 1j]])
    // >>> b = np.array([[1 + 1j, 2 + 4j],[-2 - 1j, 2 ]])
    // >>> a @ b.T
    // array([[-1.+13.j,  4. -3.j],
    //        [-3. -1.j,  0. +7.j]])

    EXPECT_EQ(C[0], -1);
    EXPECT_EQ(C[1], 4);
    EXPECT_EQ(C[2], -3);
    EXPECT_EQ(C[3], 0);
    EXPECT_EQ(C[4], 13);
    EXPECT_EQ(C[5], -3);
    EXPECT_EQ(C[6], -1);
    EXPECT_EQ(C[7], 7);
    checkCudaCall(cudaFreeHost(A));
    checkCudaCall(cudaFreeHost(B));
    checkCudaCall(cudaFreeHost(C));
}

TEST(CCGLIBTest, SimpleTestMatrixStructureDiffDimensions)
{
    // Multiply 1 x 2 matrix by 2 x 1 matrix to get 1x1 matrix.
    constexpr int n_row = 1;
    constexpr int n_col = 1;
    constexpr int n_inner = 2;
    constexpr int batch_size = 1;
    __half(*A);

    checkCudaCall(cudaMallocHost(&A, sizeof(__half) * 2 * n_row * n_inner * batch_size));

    __half(*B);

    checkCudaCall(cudaMallocHost(&B, sizeof(__half) * 2 * n_inner * n_col * batch_size));

    const int div = n_row * n_inner;
    A[0] = 1;
    A[div + 0] = 2;
    A[1] = 2;
    A[div + 1] = 1;

    B[0] = 1;
    B[div + 0] = 1;
    B[1] = 2;
    B[div + 1] = 4;

    float(*C);

    checkCudaCall(cudaMallocHost(&C, sizeof(float) * 2 * n_row * n_col * batch_size));

    ccglib_mma(A, B, C, n_row, n_col, batch_size, n_inner);
    // for debugging only.
    for (auto i = 0; i < batch_size * 2 * n_row * n_col; i++)
    {
        std::cout << i << "i" << C[i] << std::endl;
    }
    // The first matrix "A" is row-major. The second matrix "B" is column-major so
    // we end up doing a @b.T.
    // >>> import numpy as np
    // >>> a = np.array([1 + 2j, 2 + 1j])
    // >>> b = np.array([1 + 1j, 2 + 4j])
    // >>> a @ b.T
    //  array([-1.+13.j])

    EXPECT_EQ(C[0], -1);
    EXPECT_EQ(C[1], 13);

    checkCudaCall(cudaFreeHost(A));
    checkCudaCall(cudaFreeHost(B));
    checkCudaCall(cudaFreeHost(C));
}

TEST(CCGLIBTest, SimpleTestMatrixStructureDiffDimensionsBatched)
{
    // Multiply 1 x 2 matrix by 2 x 1 matrix to get 1x1 matrix.
    constexpr int n_row = 1;
    constexpr int n_col = 1;
    constexpr int n_inner = 2;
    constexpr int batch_size = 2;
    __half(*A);

    checkCudaCall(cudaMallocHost(&A, sizeof(__half) * 2 * n_row * n_inner * batch_size));

    __half(*B);

    checkCudaCall(cudaMallocHost(&B, sizeof(__half) * 2 * n_inner * n_col * batch_size));

    const int div = n_row * n_inner;
    // First matrix Re / Im
    A[0] = 1;
    A[div + 0] = 2;
    A[1] = 2;
    A[div + 1] = 1;
    // Second matrix Re / Im
    A[2 * div + 0] = 1;
    A[3 * div + 0] = 2;
    A[2 * div + 1] = 4;
    A[3 * div + 1] = 3;

    // First matrix Re / Im
    B[0] = 1;
    B[div + 0] = 1;
    B[1] = 2;
    B[div + 1] = 4;

    // Second matrix Re / Im
    B[2 * div + 0] = 1;
    B[3 * div + 0] = 0;
    B[2 * div + 1] = -1;
    B[3 * div + 1] = -4;

    float(*C);

    checkCudaCall(cudaMallocHost(&C, sizeof(float) * 2 * n_row * n_col * batch_size));

    ccglib_mma(A, B, C, n_row, n_col, batch_size, n_inner);
    // for debugging only.
    for (auto i = 0; i < batch_size * 2 * n_row * n_col; i++)
    {
        std::cout << i << "i" << C[i] << std::endl;
    }
    // The first matrix "A" is row-major. The second matrix "B" is column-major so
    // we end up doing a @ b.T.
    // >>> import numpy as np
    // >>> a = np.array([1 + 2j, 4 + 3j])
    // >>> b = np.array([1, -1 - 4j])
    // >>> a @ b.T
    //  array([9.-17.j])

    EXPECT_EQ(C[0], -1);
    EXPECT_EQ(C[1], 13);
    EXPECT_EQ(C[2], 9);
    EXPECT_EQ(C[3], -17);
    checkCudaCall(cudaFreeHost(A));
    checkCudaCall(cudaFreeHost(B));
    checkCudaCall(cudaFreeHost(C));
}


TEST(TestSpatial, TestRearrangeMatrixToCCGLIBFormat) {

   std::complex<__half> *input_matrix;
    const int n_row = 2;
    const int n_col = 2;

    __half *output_matrix;

    input_matrix = (std::complex<__half>*)malloc(n_row * n_col * sizeof(std::complex<__half>));
    output_matrix = (__half*)malloc(2 * n_row * n_col * sizeof(__half));

    input_matrix[0] = std::complex<__half>(1.0, 0);
    input_matrix[1] = std::complex<__half>(1.0, 0);
    input_matrix[2] = std::complex<__half>(1.0, 0);
    input_matrix[3] = std::complex<__half>(1.0, 0);
    

    rearrange_matrix_to_ccglib_format<__half>(input_matrix, output_matrix, n_row, n_col, true);
    
    for (int i = 0; i < n_row * n_col; ++i) {
    EXPECT_EQ(__half2float(output_matrix[i]), 1.0f);
    }

    for (int i = n_row * n_col; i < 2 * n_row * n_col; ++i) {
EXPECT_EQ(__half2float(output_matrix[i]), 0.0f);

    }
    
    free(input_matrix);
    free(output_matrix);




}


TEST(TestCCGLIBTranspose, TestTransposeSimple) {
   
    std::complex<__half> *input_matrix, *d_input;
    __half *output_matrix, *d_output;

    const int n_row = 2;
    const int n_col = 2;
    const int batch_size = 1;
    const int tile_size_x = 2;   
    const int tile_size_y = 2;
    auto precision = ccglib::Precision(ccglib::ValueType::float16).GetInputBits();
    
    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);

    input_matrix = (std::complex<__half>*)malloc(n_row * n_col * sizeof(std::complex<__half>));
    output_matrix = (__half*)malloc(n_row * n_col * 2 * sizeof(__half));

    cudaMalloc(&d_input, n_row * n_col * sizeof(std::complex<__half>));
    cudaMalloc(&d_output, n_row * n_col * 2 * sizeof(__half));
    


    for (auto i = 0; i < n_row * n_col; ++i) {
    input_matrix[i] = std::complex<__half>(__float2half(1.0f), __float2half(i * 1.0f));
                                          }
    cudaMemcpy(d_input, input_matrix, n_row * n_col * sizeof(std::complex<__half>), cudaMemcpyDefault);
    cudaDeviceSynchronize();
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    ccglib::transpose::Transpose transpose(
    batch_size, n_row, n_col, tile_size_x, tile_size_y, precision, cu_device, stream, ccglib::transpose::ComplexAxisLocation::complex_last
    );

    transpose.Run((CUdeviceptr) d_input, (CUdeviceptr) d_output);
    
    cudaMemcpyAsync(output_matrix, d_output, n_row * n_col * sizeof(__half) * 2, cudaMemcpyDefault, stream);

    cudaDeviceSynchronize();
    

    EXPECT_EQ(__half2float(output_matrix[0]), 1.0f);
    EXPECT_EQ(__half2float(output_matrix[1]), 1.0f);
    EXPECT_EQ(__half2float(output_matrix[2]), 1.0f);
    EXPECT_EQ(__half2float(output_matrix[3]), 1.0f);
    EXPECT_EQ(__half2float(output_matrix[4]), 0.0f);
    EXPECT_EQ(__half2float(output_matrix[5]), 1.0f);
    EXPECT_EQ(__half2float(output_matrix[6]), 2.0f);
    EXPECT_EQ(__half2float(output_matrix[7]), 3.0f);

    cudaStreamDestroy(stream);

}



TEST(TestSpatial, TestRearrangeCCGLIBMatrixToCompactFormat) {

   float* input_matrix;
    std::complex<float> *output_matrix;
    const int n_row = 2;
    const int n_col = 2;

    input_matrix = (float*)malloc(n_row * n_col * 2 * sizeof(float));
    output_matrix = (std::complex<float>*)malloc(n_row* n_col * sizeof(std::complex<float>));

    input_matrix[0] = 1.0f;
    input_matrix[1] = 2.0f;
    input_matrix[2] = 3.0f;
    input_matrix[3] = 4.0f;
    input_matrix[4] = 1.0f;
    input_matrix[5] = 2.0f;
    input_matrix[6] = 3.0f;
    input_matrix[7] = 4.0f;

    rearrange_ccglib_matrix_to_compact_format(input_matrix, output_matrix, n_row, n_col);

    for (int i = 0; i < n_row * n_col; ++i) {
        EXPECT_EQ(output_matrix[i], std::complex<float>(i + 1, i + 1));
    }


}


TEST(BeamformingTest, SimpleTest) {


    const int n_antennas = 2;
    const int n_samples = 2;
    const int n_beams = 1;
    std::complex<__half> *input_matrix, *weights;
    std::complex<float> *output_matrix;
    

    cudaMallocHost(&input_matrix, n_antennas*n_samples * sizeof(std::complex<__half>));
    cudaMallocHost(&output_matrix, n_beams * n_samples * sizeof(std::complex<float>));
    
    cudaMallocHost(&weights, n_antennas * n_beams * sizeof(std::complex<__half>));
    
    input_matrix[0] = std::complex<__half>(1,1);
    input_matrix[1] = std::complex<__half>(1,1);
    input_matrix[2] = std::complex<__half>(1,1);
    input_matrix[3] = std::complex<__half>(1,1);
    
    weights[0] = std::complex<__half>(0.5, 1);
    weights[1] = std::complex<__half>(0, 1);
    
    beamform<__half, float>(input_matrix, weights, output_matrix, n_antennas, n_samples, n_beams);

    for (int i = 0; i < n_beams * n_samples; ++i) {
    EXPECT_EQ(output_matrix[i], std::complex<float>(-1.5, 2.5));
    }




}

TEST(BeamformingTest, TestTwoBeams) {
    
    const int n_antennas = 2;
    const int n_samples = 2;
    const int n_beams = 2;

    std::complex<__half> *input_matrix, *weights;
    std::complex<float> *output_matrix;

    cudaMallocHost(&input_matrix, n_antennas*n_samples * sizeof(std::complex<__half>));
    cudaMallocHost(&output_matrix, n_beams * n_samples * sizeof(std::complex<float>));
    
    cudaMallocHost(&weights, n_antennas * n_beams * sizeof(std::complex<__half>));
    
    input_matrix[0] = std::complex<__half>(1, 1);
    input_matrix[1] = std::complex<__half>(1, 1);
    input_matrix[2] = std::complex<__half>(1, 1);
    input_matrix[3] = std::complex<__half>(1, 1);

    // row-major
    weights[0] = std::complex<__half>(1, 1);
    weights[1] = std::complex<__half>(2, 2);
    weights[2] = std::complex<__half>(2, 2);
    weights[3] = std::complex<__half>(4, 4);

    beamform<__half, float>(input_matrix, weights, output_matrix, n_antennas, n_samples, n_beams);

    EXPECT_EQ(output_matrix[0], std::complex<float>(0, 6));
    EXPECT_EQ(output_matrix[1], std::complex<float>(0, 6));
    EXPECT_EQ(output_matrix[2], std::complex<float>(0, 12));
    EXPECT_EQ(output_matrix[3], std::complex<float>(0, 12));

}

TEST(BeamformingTest, TestDataFormat) {

    
    const int n_antennas = 2;
    const int n_samples = 2;
    const int n_beams = 2;

    std::complex<__half> *input_matrix, *weights;
    std::complex<float> *output_matrix;

    cudaMallocHost(&input_matrix, n_antennas*n_samples * sizeof(std::complex<__half>));
    cudaMallocHost(&output_matrix, n_beams * n_samples * sizeof(std::complex<float>));
    
    cudaMallocHost(&weights, n_antennas * n_beams * sizeof(std::complex<__half>));
    // col-major  
    input_matrix[0] = std::complex<__half>(5, 5);
    input_matrix[1] = std::complex<__half>(6, 6);
    input_matrix[2] = std::complex<__half>(7, 7);
    input_matrix[3] = std::complex<__half>(8, 8);

    // row-major
    weights[0] = std::complex<__half>(1, 1);
    weights[1] = std::complex<__half>(2, 2);
    weights[2] = std::complex<__half>(3, 3);
    weights[3] = std::complex<__half>(4, 4);

    beamform<__half, float>(input_matrix, weights, output_matrix, n_antennas, n_samples, n_beams);
    //row -major
    
    //>>a = [1 + 1j, 2 + 2j; 3 + 3j, 4 + 4j]
    //
    //>>a =
    //
    //   1.0000 + 1.0000i   2.0000 + 2.0000i
    //   3.0000 + 3.0000i   4.0000 + 4.0000i

    //>>b = [5 + 5j, 7 + 7j; 6 + 6j, 8 + 8j]
    //
    //>>b =
    //
    //   5.0000 + 5.0000i   7.0000 + 7.0000i
    //   6.0000 + 6.0000i   8.0000 + 8.0000i
    //
    //>> a * b
    //
    //ans =
    //
    //   1.0e+02 *
    //
    //  0.0000 + 0.3400i   0.0000 + 0.4600i
    //  0.0000 + 0.7800i   0.0000 + 1.0600i
    EXPECT_EQ(output_matrix[0], std::complex<float>(0, 34));
    EXPECT_EQ(output_matrix[1], std::complex<float>(0, 46));
    EXPECT_EQ(output_matrix[2], std::complex<float>(0, 78));
    EXPECT_EQ(output_matrix[3], std::complex<float>(0, 106));

}

