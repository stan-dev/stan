#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>

using stan::math::check_less;
using stan::math::var;

TEST(AgradRevErrorHandlingScalar,CheckLess) {
  const char* function = "check_less";
  var x = -10.0;
  var lb = 0.0;
 
  EXPECT_TRUE(check_less(function, "x", x, lb)) 
    << "check_less should be true with x < lb";
  
  x = 1.0;
  EXPECT_THROW(check_less(function, "x", x, lb), std::domain_error)
    << "check_less should throw an exception with x > lb";

  x = lb;
  EXPECT_THROW(check_less(function, "x", x, lb), std::domain_error)
    << "check_less should throw an exception with x == lb";

  x = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less(function, "x", x, lb))
    << "check_less should be true with x == -Inf and lb = 0.0";

  x = -10.0;
  lb = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_less(function, "x", x, lb), std::domain_error)
    << "check_less should throw an exception with x == -10.0 and lb == -Inf";

  x = -std::numeric_limits<double>::infinity();
  lb = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_less(function, "x", x, lb), std::domain_error)
    << "check_less should throw an exception with x == -Inf and lb == -Inf";
  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar,CheckLess_Matrix) {
  const char* function = "check_less";
  var x;
  var high;
  Eigen::Matrix<var,Eigen::Dynamic,1> x_vec;
  Eigen::Matrix<var,Eigen::Dynamic,1> high_vec;
  x_vec.resize(3);
  high_vec.resize(3);
  
  
  // x_vec, high
  x_vec << -5, 0, 5;
  high = 10;
  EXPECT_TRUE(check_less(function, "x", x_vec, high));

  x_vec << -5, 0, 5;
  high = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less(function, "x", x_vec, high));

  x_vec << -5, 0, 5;
  high = 5;
  EXPECT_THROW(check_less(function, "x", x_vec, high),
               std::domain_error);
  
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high = 5;
  EXPECT_THROW(check_less(function, "x", x_vec, high),
               std::domain_error);

  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_less(function, "x", x_vec, high),
               std::domain_error);
  
  // x_vec, high_vec
  x_vec << -5, 0, 5;
  high_vec << 0, 5, 10;
  EXPECT_TRUE(check_less(function, "x", x_vec, high_vec));

  x_vec << -5, 0, 5;
  high_vec << std::numeric_limits<double>::infinity(), 10, 10;
  EXPECT_TRUE(check_less(function, "x", x_vec, high_vec));

  x_vec << -5, 0, 5;
  high_vec << 10, 10, 5;
  EXPECT_THROW(check_less(function, "x", x_vec, high_vec),
               std::domain_error);
  
  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high_vec << 10, 10, 10;
  EXPECT_THROW(check_less(function, "x", x_vec, high_vec), 
               std::domain_error);

  x_vec << -5, 0, std::numeric_limits<double>::infinity();
  high_vec << 10, 10, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_less(function, "x", x_vec, high_vec),
               std::domain_error);

  
  // x, high_vec
  x = -100;
  high_vec << 0, 5, 10;
  EXPECT_TRUE(check_less(function, "x", x, high_vec));

  x = 10;
  high_vec << 100, 200, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_less(function, "x", x, high_vec));

  x = 5;
  high_vec << 100, 200, 5;
  EXPECT_THROW(check_less(function, "x", x, high_vec),
               std::domain_error);
  
  x = std::numeric_limits<double>::infinity();
  high_vec << 10, 20, 30;
  EXPECT_THROW(check_less(function, "x", x, high_vec), 
               std::domain_error);

  x = std::numeric_limits<double>::infinity();
  high_vec << std::numeric_limits<double>::infinity(), 
    std::numeric_limits<double>::infinity(), 
    std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_less(function, "x", x, high_vec),
               std::domain_error);
  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckLessVarCheckUnivariate) {
  using stan::math::var;
  using stan::math::check_less;

  const char* function = "check_less";
  var a(5.0);

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_EQ(1U,stack_size);
  EXPECT_THROW(check_less(function,"a",a,2.0),std::domain_error);

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(1U,stack_size_after_call);

  EXPECT_TRUE(check_less(function,"a",a,10.0));
  stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(1U,stack_size_after_call);

  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckLessVarCheckVectorized) {
  using stan::math::var;
  using std::vector;
  using stan::math::check_less;

  int N = 5;
  const char* function = "check_less";
  vector<var> a;

  for (int i = 0; i < N; ++i)
   a.push_back(var(i));

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_EQ(5U,stack_size);
  EXPECT_TRUE(check_less(function,"a",a,10.0));

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(5U,stack_size_after_call);

  EXPECT_THROW(check_less(function,"a",a,2.0),std::domain_error);
  stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(5U,stack_size_after_call);

  stan::math::recover_memory();
}
