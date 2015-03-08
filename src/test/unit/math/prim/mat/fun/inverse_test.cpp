#include <stan/math/prim/mat/fun/inverse.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, inverse_exception) {
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::inverse;
  EXPECT_THROW(inverse(m1),std::invalid_argument);
}
