#include <stan/math/prim/mat/fun/accumulator.hpp>
#include <stan/math/prim/mat/fun/get_lp.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core/var.hpp>
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


TEST(mathMatrix,getLp) {
  using stan::math::accumulator;
  using stan::math::get_lp;
  using stan::agrad::var;

  var lp = 12.5;
  accumulator<var> lp_accum;
  EXPECT_FLOAT_EQ(12.5, get_lp(lp,lp_accum).val());

  lp_accum.add(2);
  lp_accum.add(3);
  EXPECT_FLOAT_EQ(17.5, get_lp(lp,lp_accum).val());
}

