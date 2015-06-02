#include <stan/math/prim/mat/fun/divide.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, divide) {
  stan::math::vector_d v0;
  stan::math::row_vector_d rv0;
  stan::math::matrix_d m0;

  using stan::math::divide;
  EXPECT_NO_THROW(divide(v0,2.0));
  EXPECT_NO_THROW(divide(rv0,2.0));
  EXPECT_NO_THROW(divide(m0,2.0));
}
