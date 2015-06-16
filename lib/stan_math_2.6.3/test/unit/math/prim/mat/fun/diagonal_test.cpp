#include <stan/math/prim/mat/fun/diagonal.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, diagonal) {
  stan::math::matrix_d m0;

  using stan::math::diagonal;
  EXPECT_NO_THROW(diagonal(m0));
}
