#include <stan/math/matrix/exp.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/rev/functions/exp.hpp>

TEST(AgradRevMatrix, exp_matrix) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;

  matrix_d expected_output(2,2);
  matrix_v mv(2,2), output;
  int i,j;

  mv << 1, 2, 3, 4;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val());
}
