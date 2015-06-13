#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>

using stan::math::var;
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
  stan::math::recover_memory();
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
  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckNonnegativeVarCheckUnivariate) {
  using stan::math::var;
  using stan::math::check_nonnegative;

  const char* function = "check_nonnegative";
  var a(5.0);

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_EQ(1U,stack_size);
  EXPECT_TRUE(check_nonnegative(function,"a",a));

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(1U,stack_size_after_call);

  a = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_nonnegative(function,"a",a));
  stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(2U,stack_size_after_call);

  a = 0.0;
  EXPECT_TRUE(check_nonnegative(function,"a",a));
  stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(3U,stack_size_after_call);

  a = -1.1;
  EXPECT_THROW(check_nonnegative(function,"a",a),std::domain_error);
  stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(4U,stack_size_after_call);
  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckNonnegativeVarCheckVectorized) {
  using stan::math::var;
  using std::vector;
  using stan::math::check_nonnegative;

  int N = 5;
  const char* function = "check_nonnegative";
  vector<var> a;

  for (int i = 0; i < N; ++i)
   a.push_back(var(i));

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_EQ(5U,stack_size);
  EXPECT_TRUE(check_nonnegative(function,"a",a));

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(5U,stack_size_after_call);

  a[1] = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_nonnegative(function,"a",a));
  stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(6U,stack_size_after_call);

  a[1] = -1.0;
  EXPECT_THROW(check_nonnegative(function,"a",a),std::domain_error);
  stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(7U,stack_size_after_call);

  a[1] = 0.0;
  EXPECT_TRUE(check_nonnegative(function,"a",a));
  stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(8U,stack_size_after_call);

  stan::math::recover_memory();
}
