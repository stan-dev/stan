#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <gtest/gtest.h>
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
#include <stan/math/rev/core/numeric_limits.hpp>

using stan::agrad::var;
using stan::math::check_nonnegative;

TEST(AgradRevErrorHandlingScalar,CheckNonnegative) {
  const char* function = "check_nonnegative";
  var x = 0;

  EXPECT_TRUE(check_nonnegative(function, "x", x)) 
    << "check_nonnegative should be true with finite x: " << x;

  x = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_nonnegative(function, "x", x)) 
    << "check_nonnegative should be true with x = Inf: " << x;

  x = -0.01;
  EXPECT_THROW(check_nonnegative(function, "x", x), std::domain_error) 
    << "check_nonnegative should throw exception with x = " << x;

  x = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_nonnegative(function, "x", x), std::domain_error) 
    << "check_nonnegative should throw exception with x = -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_nonnegative(function, "x", x), std::domain_error) 
    << "check_nonnegative should throw exception on NaN: " << x;
  stan::agrad::recover_memory();
}

TEST(AgradRevErrorHandlingScalar,CheckNonnegativeVectorized) {
  int N = 5;
  const char* function = "check_nonnegative";
  std::vector<var> x(N);

  x.assign(N, 0);
  EXPECT_TRUE(check_nonnegative(function, "x", x)) 
    << "check_nonnegative(vector) should be true with finite x: " << x[0];

  x.assign(N, std::numeric_limits<double>::infinity());
  EXPECT_TRUE(check_nonnegative(function, "x", x)) 
    << "check_nonnegative(vector) should be true with x = Inf: " << x[0];

  x.assign(N, -0.01);
  EXPECT_THROW(check_nonnegative(function, "x", x), std::domain_error) 
    << "check_nonnegative should throw exception with x = " << x[0];


  x.assign(N, -std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_nonnegative(function, "x", x), std::domain_error) 
    << "check_nonnegative(vector) should throw an exception with x = -Inf: " << x[0];

  x.assign(N, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(check_nonnegative(function, "x", x), std::domain_error) 
    << "check_nonnegative(vector) should throw exception on NaN: " << x[0];
  stan::agrad::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckNonnegativeVarCheckUnivariate) {
  using stan::agrad::var;
  using stan::math::check_nonnegative;

  const char* function = "check_nonnegative";
  var a(5.0);

  size_t stack_size = stan::agrad::ChainableStack::var_stack_.size();

  EXPECT_EQ(1U,stack_size);
  EXPECT_TRUE(check_nonnegative(function,"a",a));

  size_t stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(1U,stack_size_after_call);

  a = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_nonnegative(function,"a",a));
  stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(2U,stack_size_after_call);

  a = 0.0;
  EXPECT_TRUE(check_nonnegative(function,"a",a));
  stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(3U,stack_size_after_call);

  a = -1.1;
  EXPECT_THROW(check_nonnegative(function,"a",a),std::domain_error);
  stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(4U,stack_size_after_call);
  stan::agrad::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckNonnegativeVarCheckVectorized) {
  using stan::agrad::var;
  using std::vector;
  using stan::math::check_nonnegative;

  int N = 5;
  const char* function = "check_nonnegative";
  vector<var> a;

  for (int i = 0; i < N; ++i)
   a.push_back(var(i));

  size_t stack_size = stan::agrad::ChainableStack::var_stack_.size();

  EXPECT_EQ(5U,stack_size);
  EXPECT_TRUE(check_nonnegative(function,"a",a));

  size_t stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(5U,stack_size_after_call);

  a[1] = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_nonnegative(function,"a",a));
  stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(6U,stack_size_after_call);

  a[1] = -1.0;
  EXPECT_THROW(check_nonnegative(function,"a",a),std::domain_error);
  stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(7U,stack_size_after_call);

  a[1] = 0.0;
  EXPECT_TRUE(check_nonnegative(function,"a",a));
  stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(8U,stack_size_after_call);

  stan::agrad::recover_memory();
}
