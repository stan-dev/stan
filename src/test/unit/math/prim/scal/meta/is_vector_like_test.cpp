#include <stan/math/prim/scal/meta/is_vector_like.hpp>
#include <gtest/gtest.h>

TEST(is_vector_like, double) {
  EXPECT_FALSE(stan::is_vector_like<double>::value);
}

TEST(is_vector_like, double_pointer) {
  EXPECT_TRUE(stan::is_vector_like<double *>::value);
}
