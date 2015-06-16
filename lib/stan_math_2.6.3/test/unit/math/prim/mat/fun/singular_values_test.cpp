#include <stan/math/prim/mat/fun/singular_values.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, singular_values) {
  stan::math::matrix_d m0(1,1);
  m0 << 1.0;

  using stan::math::singular_values;
  EXPECT_NO_THROW(singular_values(m0));
}
