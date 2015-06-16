#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/scal/err/check_greater_or_equal.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>

using stan::math::var;
using stan::math::check_greater_or_equal;

TEST(AgradRevErrorHandlingScalar,CheckGreaterOrEqual) {
  const char* function = "check_greater_or_equal";
  var x = 10.0;
  var lb = 0.0;
 
  EXPECT_TRUE(check_greater_or_equal(function, "x", x, lb)) 
    << "check_greater_or_equal should be true with x > lb";
  
  x = -1.0;
  EXPECT_THROW(check_greater_or_equal(function, "x", x, lb),
               std::domain_error)
    << "check_greater_or_equal should throw an exception with x < lb";

  x = lb;
  EXPECT_NO_THROW(check_greater_or_equal(function, "x", x, lb))
    << "check_greater_or_equal should not throw an exception with x == lb";

  x = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, "x", x, lb))
    << "check_greater should be true with x == Inf and lb = 0.0";

  x = 10.0;
  lb = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater_or_equal(function, "x", x, lb),
               std::domain_error)
    << "check_greater should throw an exception with x == 10.0 and lb == Inf";

  x = std::numeric_limits<double>::infinity();
  lb = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(check_greater_or_equal(function, "x", x, lb))
    << "check_greater should not throw an exception with x == Inf and lb == Inf";
  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar,CheckGreaterOrEqualMatrix) {
  const char* function = "check_greater_or_equal";
  var x;
  var low;
  Eigen::Matrix<var,Eigen::Dynamic,1> x_vec;
  Eigen::Matrix<var,Eigen::Dynamic,1> low_vec;
  x_vec.resize(3);
  low_vec.resize(3);

  // x_vec, low_vec
  x_vec   << -1, 0, 1;
  low_vec << -2, -1, 0;
  EXPECT_TRUE(check_greater_or_equal(function, "x", x_vec, low_vec)) 
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>";

  x_vec   <<   -1,    0,   1;
  low_vec << -1.1, -0.1, 0.9;
  EXPECT_TRUE(check_greater_or_equal(function, "x", x_vec, low_vec)) 
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>";


  x_vec   << -1, 0, std::numeric_limits<double>::infinity();
  low_vec << -2, -1, 0;
  EXPECT_TRUE(check_greater_or_equal(function, "x", x_vec, low_vec)) 
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, y has infinity";
  
  x_vec   << -1, 0, 1;
  low_vec << -2, 0, 0;
  EXPECT_TRUE(check_greater_or_equal(function, "x", x_vec, low_vec))
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, should pass for index 1";
  
  x_vec   << -1, 0,  1;
  low_vec << -2, -1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater_or_equal(function, "x", x_vec, low_vec), 
               std::domain_error) 
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, should fail with infinity";
  
  x_vec   << -1, 0,  std::numeric_limits<double>::infinity();
  low_vec << -2, -1, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, "x", x_vec, low_vec))
    << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, both bound and value infinity";
  
  x_vec   << -1, 0,  1;
  low_vec << -2, -1, -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, "x", x_vec, low_vec))
  << "check_greater_or_equal: matrix<3,1>, matrix<3,1>, should pass with -infinity";

  // x_vec, low
  x_vec   << -1, 0, 1;
  low = -2;
  EXPECT_TRUE(check_greater_or_equal(function, "x", x_vec, low)) 
    << "check_greater_or_equal: matrix<3,1>, double";

  x_vec   <<   -1,    0,   1;
  low = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, "x", x_vec, low)) 
    << "check_greater_or_equal: matrix<3,1>, double";

  x_vec   << -1, 0, 1;
  low = 0;
  EXPECT_THROW(check_greater_or_equal(function, "x", x_vec, low),
               std::domain_error) 
    << "check_greater_or_equal: matrix<3,1>, double, should fail for index 1/2";
  
  x_vec   << -1, 0,  1;
  low = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater_or_equal(function, "x", x_vec, low), 
               std::domain_error) 
    << "check_greater_or_equal: matrix<3,1>, double, should fail with infinity";
  
  // x, low_vec
  x = 2;
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater_or_equal(function, "x", x, low_vec)) 
    << "check_greater_or_equal: double, matrix<3,1>";

  x = 10;
  low_vec << -1, 0, -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, "x", x, low_vec)) 
    << "check_greater_or_equal: double, matrix<3,1>, low has -inf";

  x = 10;
  low_vec << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_greater_or_equal(function, "x", x, low_vec), 
               std::domain_error) 
    << "check_greater_or_equal: double, matrix<3,1>, low has inf";
  
  x = std::numeric_limits<double>::infinity();
  low_vec << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_greater_or_equal(function, "x", x, low_vec))
    << "check_greater_or_equal: double, matrix<3,1>, x is inf, low has inf";
  
  x = std::numeric_limits<double>::infinity();
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater_or_equal(function, "x", x, low_vec)) 
    << "check_greater_or_equal: double, matrix<3,1>, x is inf";

  x = 1.1;
  low_vec << -1, 0, 1;
  EXPECT_TRUE(check_greater_or_equal(function, "x", x, low_vec)) 
    << "check_greater_or_equal: double, matrix<3,1>";
  
  x = 0.9;
  low_vec << -1, 0, 1;
  EXPECT_THROW(check_greater_or_equal(function, "x", x, low_vec), 
               std::domain_error) 
    << "check_greater_or_equal: double, matrix<3,1>";
  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckGreaterOrEqualVarCheckUnivariate) {
  using stan::math::var;
  using stan::math::check_greater_or_equal;

  const char* function = "check_greater_or_equal";
  var a(5.0);

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_EQ(1U,stack_size);
  EXPECT_TRUE(check_greater_or_equal(function,"a",a,2.0));

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(1U,stack_size_after_call);

  EXPECT_THROW(check_greater_or_equal(function,"a",a,10.0),std::domain_error);
  stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(1U,stack_size_after_call);

  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckFiniteVarCheckVectorized) {
  using stan::math::var;
  using std::vector;
  using stan::math::check_greater_or_equal;

  int N = 5;
  const char* function = "check_greater_or_equal";
  vector<var> a;

  for (int i = 0; i < N; ++i)
   a.push_back(var(i));

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_EQ(5U,stack_size);
  EXPECT_TRUE(check_greater_or_equal(function,"a",a,-1.0));

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(5U,stack_size_after_call);

  EXPECT_THROW(check_greater_or_equal(function,"a",a,2.0),std::domain_error);
  stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(5U,stack_size_after_call);

  stan::math::recover_memory();
}
