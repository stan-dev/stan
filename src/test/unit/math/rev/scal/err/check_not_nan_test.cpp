#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/core/numeric_limits.hpp>
#include <stan/math/rev/core/operator_addition.hpp>
#include <stan/math/rev/core/operator_divide_equal.hpp>
#include <stan/math/rev/core/operator_division.hpp>
#include <stan/math/rev/core/operator_equal.hpp>
#include <stan/math/rev/core/operator_greater_than.hpp>
#include <stan/math/rev/core/operator_greater_than_or_equal.hpp>
#include <stan/math/rev/core/operator_less_than.hpp>
#include <stan/math/rev/core/operator_less_than_or_equal.hpp>
#include <stan/math/rev/core/operator_minus_equal.hpp>
#include <stan/math/rev/core/operator_multiplication.hpp>
#include <stan/math/rev/core/operator_multiply_equal.hpp>
#include <stan/math/rev/core/operator_not_equal.hpp>
#include <stan/math/rev/core/operator_plus_equal.hpp>
#include <stan/math/rev/core/operator_subtraction.hpp>
#include <stan/math/rev/core/operator_unary_decrement.hpp>
#include <stan/math/rev/core/operator_unary_increment.hpp>
#include <stan/math/rev/core/operator_unary_negative.hpp>
#include <stan/math/rev/core/operator_unary_not.hpp>
#include <stan/math/rev/core/operator_unary_plus.hpp>

TEST(AgradRevErrorHandlingScalar,CheckNotNan) {
  using stan::agrad::var;
  using stan::math::check_not_nan;
  const char* function = "check_not_nan";

  var x = 0;
  double x_d = 0;
 
  EXPECT_TRUE(check_not_nan(function, "x", x))
    << "check_not_nan should be true with finite x: " << x;
  EXPECT_TRUE(check_not_nan(function, "x", x_d))
    << "check_not_nan should be true with finite x: " << x_d;
  
  x = std::numeric_limits<var>::infinity();
  x_d = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, "x", x))
    << "check_not_nan should be true with x = Inf: " << x;
  EXPECT_TRUE(check_not_nan(function, "x", x_d))
    << "check_not_nan should be true with x = Inf: " << x_d;

  x = -std::numeric_limits<var>::infinity();
  x_d = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_not_nan(function, "x", x))
    << "check_not_nan should be true with x = -Inf: " << x;
  EXPECT_TRUE(check_not_nan(function, "x", x_d))
    << "check_not_nan should be true with x = -Inf: " << x_d;

  x = std::numeric_limits<var>::quiet_NaN();
  x_d = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_not_nan(function, "x", x), std::domain_error)
    << "check_not_nan should throw exception on NaN: " << x;
  EXPECT_THROW(check_not_nan(function, "x", x_d), std::domain_error)
    << "check_not_nan should throw exception on NaN: " << x_d;
  stan::agrad::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckNotNanVarCheckUnivariate) {
  using stan::agrad::var;
  using stan::math::check_not_nan;

  const char* function = "check_not_nan";
  var a(5.0);

  size_t stack_size = stan::agrad::ChainableStack::var_stack_.size();

  EXPECT_EQ(1U,stack_size);
  EXPECT_TRUE(check_not_nan(function,"a",a));

  size_t stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(1U,stack_size_after_call);

  stan::agrad::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckNotNanVarCheckVectorized) {
  using stan::agrad::var;
  using std::vector;
  using stan::math::check_not_nan;

  int N = 5;
  const char* function = "check_not_nan";
  vector<var> a;

  for (int i = 0; i < N; ++i)
   a.push_back(var(i));

  size_t stack_size = stan::agrad::ChainableStack::var_stack_.size();

  EXPECT_EQ(5U,stack_size);
  EXPECT_TRUE(check_not_nan(function,"a",a));

  size_t stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(5U,stack_size_after_call);
  stan::agrad::recover_memory();
}
