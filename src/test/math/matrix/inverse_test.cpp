#include <stan/math/matrix/inverse.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, inverse_exception) {
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::inverse;
  EXPECT_THROW(inverse(m1),std::domain_error);
}
