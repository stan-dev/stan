#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/agrad/rev.hpp>
#include <gtest/gtest.h>

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
