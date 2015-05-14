#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/exp.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>

TEST(AgradRevMatrix, exp_matrix) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::math::matrix_v;

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
