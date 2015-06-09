#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>

TEST(AgradRevErrorHandlingScalar,CheckNotNan) {
  using stan::math::var;
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
  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckNotNanVarCheckUnivariate) {
  using stan::math::var;
  using stan::math::check_not_nan;

  const char* function = "check_not_nan";
  var a(5.0);

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_EQ(1U,stack_size);
  EXPECT_TRUE(check_not_nan(function,"a",a));

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(1U,stack_size_after_call);

  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckNotNanVarCheckVectorized) {
  using stan::math::var;
  using std::vector;
  using stan::math::check_not_nan;

  int N = 5;
  const char* function = "check_not_nan";
  vector<var> a;

  for (int i = 0; i < N; ++i)
   a.push_back(var(i));

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_EQ(5U,stack_size);
  EXPECT_TRUE(check_not_nan(function,"a",a));

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(5U,stack_size_after_call);
  stan::math::recover_memory();
}


TEST(ErrorHandlingScalar, CheckNotNanVarCheckUnivariate) {
  using stan::math::var;
  using stan::math::check_not_nan;
  
  const char* function = "check_not_nan";
  var a(5.0);

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_TRUE(1U == stack_size);
  EXPECT_TRUE(check_not_nan(function,"a",a));

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_TRUE(1U == stack_size_after_call);

  stan::math::recover_memory();
}

TEST(ErrorHandlingScalar, CheckNotNanVarCheckVectorized) {
  using stan::math::var;
  using std::vector;
  using stan::math::check_not_nan;

  int N = 5;
  const char* function = "check_not_nan";
  vector<var> a;

  for (int i = 0; i < N; ++i)
   a.push_back(var(i));

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_TRUE(5U == stack_size);
  EXPECT_TRUE(check_not_nan(function,"a",a));

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_TRUE(5U == stack_size_after_call);
  stan::math::recover_memory();
}
