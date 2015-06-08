#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/scal/err/check_equal.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>

using stan::math::check_equal;
using stan::math::var;

TEST(AgradRevErrorHandlingScalar,CheckEqual) {
  const char* function = "check_equal";
  var x = 0.0;
  var eq = 0.0;
 
  EXPECT_TRUE(check_equal(function, "x", x, eq))
    << "check_equal should be true with x = eq";
  
  x = -1.0;
  EXPECT_THROW(check_equal(function, "x", x, eq),
               std::domain_error)
    << "check_equal should throw an exception with x < eq";

  x = eq;
  EXPECT_NO_THROW(check_equal(function, "x", x, eq))
    << "check_equal should not throw an exception with x == eq";

  x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_equal(function, "x", x, eq), 
               std::domain_error)
    << "check_equal should be false with x == Inf and eq = 0.0";

  x = 10.0;
  eq = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_equal(function, "x", x, eq),
               std::domain_error)
    << "check_equal should throw an exception with x == 10.0 and eq == Inf";

  x = std::numeric_limits<double>::infinity();
  eq = std::numeric_limits<double>::infinity();
  EXPECT_NO_THROW(check_equal(function, "x", x, eq))
    << "check_equal should not throw an exception with x == Inf and eq == Inf";
  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar,CheckEqualMatrix) {
  const char* function = "check_equal";
  Eigen::Matrix<var,Eigen::Dynamic,1> x_vec;
  Eigen::Matrix<var,Eigen::Dynamic,1> eq_vec;
  x_vec.resize(3);
  eq_vec.resize(3);

  // x_vec, low_vec
  x_vec   << -1, 0, 1;
  eq_vec << -1, 0, 1;
  EXPECT_TRUE(check_equal(function, "x", x_vec, eq_vec)) 
    << "check_equal: matrix<3,1>, matrix<3,1>";

  x_vec   <<   -1,    0,   1;
  eq_vec << -1.1, -0.1, 0.9;
  EXPECT_THROW(check_equal(function, "x", x_vec, eq_vec),
               std::domain_error) 
    << "check_equal: matrix<3,1>, matrix<3,1>";
  
  x_vec   << -1, 0,  1;
  eq_vec << -2, -1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_equal(function, "x", x_vec, eq_vec), 
               std::domain_error) 
    << "check_equal: matrix<3,1>, matrix<3,1>, should fail with infinity";
  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckEqualVarCheckUnivariate) {

  const char* function = "check_equal";
  var a(5.0);
  var b(4.0);

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_EQ(2U,stack_size);
  EXPECT_THROW(check_equal(function,"a",a,b),std::domain_error);

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(2U,stack_size_after_call);

  b = 5.0;
  EXPECT_TRUE(check_equal(function,"a",a,b));
  stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(3U,stack_size_after_call);

  b = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_equal(function,"a",a,b),std::domain_error);
  stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(4U,stack_size_after_call);
  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckNotNanVarCheckVectorized) {
  using std::vector;

  int N = 5;
  const char* function = "check_not_nan";
  vector<var> a;
  vector<var> b;

  for (int i = 0; i < N; ++i){
   a.push_back(var(i));
   b.push_back(var(i));
  }

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_EQ(10U,stack_size);
  EXPECT_TRUE(check_equal(function,"a",a,b));

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(10U,stack_size_after_call);

  b[1] = 4.0;
  EXPECT_THROW(check_equal(function,"a",a,b),std::domain_error);
  stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(11U,stack_size_after_call);

  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckEqualVarCheckMatrix) {
  using Eigen::Matrix;
  using Eigen::Dynamic;

  int N = 2;
  const char* function = "check_not_nan";
  Matrix<var,Dynamic,Dynamic> a(N,N);
  Matrix<var,Dynamic,Dynamic> b(N,N);

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j){
      a(i,j) = var(i+j+1);
      b(i,j) = var(i+j+1);
    }

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();
  size_t stack_size_expected = 2 * N * N;

  EXPECT_EQ(stack_size_expected,stack_size);
  EXPECT_TRUE(check_equal(function,"a",a,b));

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(stack_size_expected,stack_size_after_call);

  b(1,1) = 45;
  EXPECT_THROW(check_equal(function,"a",a,b),std::domain_error);
  stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(stack_size_expected + 1,stack_size_after_call);

  stan::math::recover_memory();
}
