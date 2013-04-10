#include <stan/math/matrix/minus.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, minus) {
  stan::math::vector_d v0;
  stan::math::row_vector_d rv0;
  stan::math::matrix_d m0;

  EXPECT_EQ(0,stan::math::minus(v0).size());
  EXPECT_EQ(0,stan::math::minus(rv0).size());
  EXPECT_EQ(0,stan::math::minus(m0).size());
}
