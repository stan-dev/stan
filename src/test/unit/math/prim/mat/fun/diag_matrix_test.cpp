#include <stan/math/prim/mat/fun/diag_matrix.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, inverse_exception) {
  stan::math::vector_d v0;

  using stan::math::diag_matrix;
  EXPECT_NO_THROW(diag_matrix(v0));
}
