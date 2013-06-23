#include <stan/math/matrix/exp.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, exp) {
  using stan::math::exp;
  stan::math::matrix_d expected_output(2,2);
  stan::math::matrix_d mv(2,2), output;
  int i,j;

  mv << 1, 2, 3, 4;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j));
}
