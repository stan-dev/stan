#include <stan/math/matrix/diag_matrix.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, inverse_exception) {
  stan::math::vector_d v0;

  using stan::math::diag_matrix;
  EXPECT_NO_THROW(diag_matrix(v0));
}
