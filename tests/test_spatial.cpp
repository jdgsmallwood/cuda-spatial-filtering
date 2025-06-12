#include <gtest/gtest.h>
#include "spatial/spatial.hpp"
#include <vector>
#include <cuComplex.h>
#include <cuda_runtime.h>

TEST(MyLibTest, AddFunction)
{
    EXPECT_EQ(add(2, 3), 5);
    EXPECT_EQ(add(-1, 1), 0);
}

TEST(MyLibTest, IncrementArrayCUDA)
{
    int data[5] = {1, 2, 3, 4, 5};
    incrementArray(data, 5);
    EXPECT_EQ(data[0], 2);
    EXPECT_EQ(data[4], 6);
}

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
    (*samples)[1][1][4][0][0] = Sample(2, 3);
    (*samples)[1][1][5][0][0] = Sample(4, 5);
    std::cout << "Creating visibilities" << std::endl;
    Visibilities *visibilities;
    checkCudaCall(cudaMallocHost(&visibilities, sizeof(Visibilities)));
    std::cout << "Starting correlation..." << std::endl; 
    correlate(samples, visibilities);
    std::cout << "Finished correlation..." << std::endl;
    
    print_nonzero_visibilities(visibilities);
    
    EXPECT_EQ((*visibilities)[2][19][0][0], Visibility(23, 2));
    EXPECT_EQ((*visibilities)[2][14][0][0], Visibility(13, 0));
    EXPECT_EQ((*visibilities)[2][20][0][0], Visibility(41,0));

    checkCudaCall(cudaFreeHost(visibilities));

    checkCudaCall(cudaFreeHost(samples));
    } catch (std::exception &error) {
        std::cerr << error.what() << std::endl;
    }
}