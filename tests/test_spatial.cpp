#include <gtest/gtest.h>
#include "spatial/spatial.hpp"
#include <vector>
#include <cuComplex.cu>

TEST(MyLibTest, AddFunction) {
    EXPECT_EQ(add(2, 3), 5);
    EXPECT_EQ(add(-1, 1), 0);
}

TEST(MyLibTest, IncrementArrayCUDA) {
    int data[5] = {1, 2, 3, 4, 5};
    incrementArray(data, 5);
    EXPECT_EQ(data[0], 2);
    EXPECT_EQ(data[4], 6);
}

TEST(EigenvalueDecompositionTest, SimpleValueTest) {

 const std::vector<u_data_type> A = {
    make_cuComplex(3.5,0.0), make_cuComplex(0.5, 1.0), make_cuComplex(0.0, 0.0), 
    make_cuComplex(0.5, -1.0), make_cuComplex(3.5, 0.0), make_cuComplex(0.0, 0.0), 
    make_cuComplex(0.0, 0.0), make_cuComplex(0.0,0.0), make_cuComplex(2.0, 0)};   

float* h_eigenvalues;
cudaMallocHost((void**)&h_eigenvalues, 3 * sizeof(float));

eigendecomposition<cuComplex>(h_eigenvalues);

EXPECT_EQ(h_eigenvalues[0], 2.0);
EXPECT_EQ(h_eigenvalues[1], 2.4);
EXPECT_EQ(h_eigenvalues[2], 4.7);


}