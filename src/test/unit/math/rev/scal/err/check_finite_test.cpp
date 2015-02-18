#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/rev/core/recover_memory.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>

TEST(AgradRevErrorHandlingScalar,CheckFinite) {
  using stan::agrad::var;
  using stan::math::check_finite;
 
  const char* function = "check_bounded";
  const char* name = "x";
  var x = 0;
 
  EXPECT_TRUE(check_finite(function, name, x)) 
    << "check_finite should be TRUE with x: " << x;
  
  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_finite(function, name, x), std::domain_error) 
    << "check_finite should throw with x: " << x;

  x = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_finite(function, name, x), std::domain_error) 
    << "check_finite should throw with x: " << x;

  x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_finite(function, name, x), std::domain_error) 
    << "check_finite should throw with x: " << x;
  stan::agrad::recover_memory();
}


TEST(AgradRevErrorHandlingScalar, CheckFiniteVarCheckUnivariate) {
  using stan::agrad::var;
  using stan::math::check_finite;

  const char* function = "check_finite";
  var a(5.0);

  size_t stack_size = stan::agrad::ChainableStack::var_stack_.size();

  EXPECT_EQ(1U,stack_size);
  EXPECT_TRUE(check_finite(function,"a",a));

  size_t stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(1U,stack_size_after_call);

  a = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_finite(function,"a",a),std::domain_error);
  stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(2U,stack_size_after_call);

  stan::agrad::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckFiniteVarCheckVectorized) {
  using stan::agrad::var;
  using std::vector;
  using stan::math::check_finite;

  int N = 5;
  const char* function = "check_finite";
  vector<var> a;

  for (int i = 0; i < N; ++i)
   a.push_back(var(i));

  size_t stack_size = stan::agrad::ChainableStack::var_stack_.size();

  EXPECT_EQ(5U,stack_size);
  EXPECT_TRUE(check_finite(function,"a",a));

  size_t stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(5U,stack_size_after_call);

  a[1] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_finite(function,"a",a),std::domain_error);
  stack_size_after_call = stan::agrad::ChainableStack::var_stack_.size();
  EXPECT_EQ(6U,stack_size_after_call);

  stan::agrad::recover_memory();
}
