#include <stan/math/prim/mat/fun/log.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

// log tests
TEST(MathMatrix, log) {
  using stan::math::log;
  stan::math::matrix_d expected_output(2,2);
  stan::math::matrix_d mv(2,2), output;
  int i,j;

  mv << 1, 2, 3, 4;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j));
}
