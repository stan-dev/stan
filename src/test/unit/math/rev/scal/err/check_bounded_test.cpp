#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core/recover_memory.hpp>
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
#include <stan/math/rev/core/std_numeric_limits.hpp>

TEST(AgradRevErrorHandlingScalar,CheckBounded_X) {
  using stan::agrad::var;
  using stan::math::check_bounded;
 
  const char* function = "check_bounded";
  const char* name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
 
  EXPECT_TRUE(check_bounded(function, name, x, low, high)) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  x = low;
  EXPECT_TRUE(check_bounded(function, name, x, low, high)) 
    << "check_bounded should be TRUE with x: " << x << " equal to the lower bound: " << low;

  x = high;
  EXPECT_TRUE(check_bounded(function, name, x, low, high)) 
    << "check_bounded should be TRUE with x: " << x << " equal to the lower bound: " << low;

  x = low-1;
  EXPECT_THROW(check_bounded(function, name, x, low, high), 
               std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;  
  
  x = high+1;
  EXPECT_THROW(check_bounded(function, name, x, low, high), 
               std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = std::numeric_limits<var>::quiet_NaN();
  EXPECT_THROW(check_bounded(function, name, x, low, high), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = -std::numeric_limits<var>::infinity();
  EXPECT_THROW(check_bounded(function, name, x, low, high), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = std::numeric_limits<var>::infinity();
  EXPECT_THROW(check_bounded(function, name, x, low, high), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;
  stan::agrad::recover_memory();
}

TEST(AgradRevErrorHandlingScalar,CheckBounded_Low) {
  using stan::agrad::var;
  using stan::math::check_bounded;

  const char* function = "check_bounded";
  const char* name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
 
  EXPECT_TRUE(check_bounded(function, name, x, low, high)) 
    << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;
  
  low = -std::numeric_limits<var>::infinity();
  EXPECT_TRUE(check_bounded(function, name, x, low, high)) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<var>::quiet_NaN();
  EXPECT_THROW(check_bounded(function, name, x, low, high), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<var>::infinity();
  EXPECT_THROW(check_bounded(function, name, x, low, high), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
  stan::agrad::recover_memory();
}
TEST(AgradRevErrorHandlingScalar,CheckBounded_High) {
  using stan::agrad::var;
  using stan::math::check_bounded;

  const char* function = "check_bounded";
  const char* name = "x";
  var x = 0;
  var low = -1;
  var high = 1;
 
  EXPECT_TRUE(check_bounded(function, name, x, low, high)) 
    << "check_bounded should be true x: " << x << " and bounds: " << low << ", " << high;

  high = std::numeric_limits<var>::infinity();
  EXPECT_TRUE(check_bounded(function, name, x, low, high)) 
    << "check_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  high = std::numeric_limits<var>::quiet_NaN();
  EXPECT_THROW(check_bounded(function, name, x, low, high), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  high = -std::numeric_limits<var>::infinity();
  EXPECT_THROW(check_bounded(function, name, x, low, high), std::domain_error) 
    << "check_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
  stan::agrad::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckBoundedVarCheckUnivariate) {
  using stan::agrad::var;
  using stan::math::check_bounded;

  const char* function = "check_bounded";
  var a(5.0);

  size_t stack_size = stan::agrad::ChainableStack::var_stack_.size();

  EXPECT_EQ(1U,stack_size);
  EXPECT_TRUE(check_bounded(function,"a",a,4.0,6.0));

  size_t stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(1U,stack_size_after_call);

  stan::agrad::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckBoundedVarCheckVectorized) {
  using stan::agrad::var;
  using std::vector;
  using stan::math::check_bounded;

  int N = 5;
  const char* function = "check_bounded";
  vector<var> a;

  for (int i = 0; i < N; ++i)
   a.push_back(var(i));

  size_t stack_size = stan::agrad::ChainableStack::var_stack_.size();

  EXPECT_EQ(5U,stack_size);
  EXPECT_TRUE(check_bounded(function,"a",a,-1.0,6.0));

  size_t stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(5U,stack_size_after_call);
  stan::agrad::recover_memory();
}
