#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/rev/core.hpp>
#include <gtest/gtest.h>

TEST(AgradRevErrorHandlingScalar, checkConsistentSizes) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::check_consistent_sizes;
  using stan::size_of;
  using stan::math::var;

  const char* function = "testConsSizes";
  const char* name1 = "name1";
  const char* name2 = "name2";
  const char* name3 = "name3";
  const char* name4 = "name4";
  

  Matrix<var,Dynamic,1> v1(4);
  Matrix<var,Dynamic,1> v2(4);
  Matrix<var,Dynamic,1> v3(4);
  Matrix<var,Dynamic,1> v4(4);
  ASSERT_EQ(4U, size_of(v1));
  ASSERT_EQ(4U, size_of(v2));
  ASSERT_EQ(4U, size_of(v3));
  ASSERT_EQ(4U, size_of(v4));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v2));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v2, name3, v3));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v2, name3, v3, name4, v4));
  
  Matrix<var,Dynamic,1> v(3);
  
  ASSERT_EQ(3U, size_of(v));
  const char* name = "inconsistent";
  EXPECT_THROW(check_consistent_sizes(function, name, v, name2, v2),
               std::invalid_argument);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name, v),
               std::invalid_argument);
  EXPECT_THROW(check_consistent_sizes(function, name, v, name2, v2, name3, v3),
               std::invalid_argument);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name, v, name3, v3),
               std::invalid_argument);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name2, v2, name, v),
               std::invalid_argument);

  EXPECT_THROW(check_consistent_sizes(function, name, v, name2, v2, name3, v3, name4, v4),
               std::invalid_argument);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name, v, name3, v3, name4, v4),
               std::invalid_argument);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name2, v2, name, v, name4, v4),
               std::invalid_argument);
  EXPECT_THROW(check_consistent_sizes(function, name, v, name2, v2, name3, v3, name, v),
               std::invalid_argument);
  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, CheckConsistentSizeVarCheckVectorized) {
  using stan::math::var;
  using std::vector;
  using stan::math::check_consistent_sizes;

  int N = 5;
  const char* function = "check_consistent_size";
  vector<var> a;
  vector<var> b;

  for (int i = 0; i < N; ++i){
   b.push_back(var(i+1));
   a.push_back(var(i));
  }

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_EQ(10U,stack_size);
  EXPECT_TRUE(check_consistent_sizes(function,"a",a,"b",b));

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(10U,stack_size_after_call);
  stan::math::recover_memory();
}
