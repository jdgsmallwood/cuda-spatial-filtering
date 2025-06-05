#pragma once
#include <vector>

int add(int a, int b);
void incrementArray(int *data, int size);
template <typename T>
void eigendecomposition(float *h_eigenvalues, int n, const std::vector<T> *A);
void correlate();