#include <gtest/gtest.h>
#include "spatial/spatial.hpp"

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