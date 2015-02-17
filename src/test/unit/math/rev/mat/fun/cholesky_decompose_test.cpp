#include <stan/math/prim/mat/fun/cholesky_decompose.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/rev/core/var.hpp>
#include <stan/math/prim/mat/fun/singular_values.hpp>
#include <stan/math/prim/mat/fun/transpose.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/multiply.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/operator_addition.hpp>
#include <stan/math/rev/scal/fun/operator_divide_equal.hpp>
#include <stan/math/rev/scal/fun/operator_division.hpp>
#include <stan/math/rev/scal/fun/operator_equal.hpp>
#include <stan/math/rev/scal/fun/operator_greater_than.hpp>
#include <stan/math/rev/scal/fun/operator_greater_than_or_equal.hpp>
#include <stan/math/rev/scal/fun/operator_less_than.hpp>
#include <stan/math/rev/scal/fun/operator_less_than_or_equal.hpp>
#include <stan/math/rev/scal/fun/operator_minus_equal.hpp>
#include <stan/math/rev/scal/fun/operator_multiplication.hpp>
#include <stan/math/rev/scal/fun/operator_multiply_equal.hpp>
#include <stan/math/rev/scal/fun/operator_not_equal.hpp>
#include <stan/math/rev/scal/fun/operator_plus_equal.hpp>
#include <stan/math/rev/scal/fun/operator_subtraction.hpp>
#include <stan/math/rev/scal/fun/operator_unary_decrement.hpp>
#include <stan/math/rev/scal/fun/operator_unary_increment.hpp>
#include <stan/math/rev/scal/fun/operator_unary_negative.hpp>
#include <stan/math/rev/scal/fun/operator_unary_not.hpp>
#include <stan/math/rev/scal/fun/operator_unary_plus.hpp>
#include <stan/math/rev/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/rev/scal/meta/numeric_limits.hpp>

TEST(AgradRevMatrix,mat_cholesky) {
  using stan::agrad::matrix_v;
  using stan::math::transpose;
  using stan::math::cholesky_decompose;
  using stan::math::singular_values;

  // symmetric
  matrix_v X(2,2);
  AVAR a = 3.0;
  AVAR b = -1.0;
  AVAR c = -1.0;
  AVAR d = 1.0;
  X << a, b, 
    c, d;

  matrix_v L = cholesky_decompose(X);

  matrix_v LL_trans = multiply(L,transpose(L));
  EXPECT_FLOAT_EQ(a.val(),LL_trans(0,0).val());
  EXPECT_FLOAT_EQ(b.val(),LL_trans(0,1).val());
  EXPECT_FLOAT_EQ(c.val(),LL_trans(1,0).val());
  EXPECT_FLOAT_EQ(d.val(),LL_trans(1,1).val());

  EXPECT_NO_THROW(singular_values(X));
}

