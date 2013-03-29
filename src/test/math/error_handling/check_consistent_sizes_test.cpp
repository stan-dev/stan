#include <stan/math/error_handling/check_consistent_sizes.hpp>
#include <gtest/gtest.h>
#include <stan/math/error_handling/default_policy.hpp>

using stan::math::default_policy;

TEST(MathMatrixErrorHandling, checkConsistentSizes) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::check_consistent_sizes;
  using stan::size_of;

  const char* function = "testConsSizes(%1%)";
  const char* name1 = "name1";
  const char* name2 = "name2";
  const char* name3 = "name3";
  
  double result;

  Matrix<double,Dynamic,1> v1(4);
  Matrix<double,Dynamic,1> v2(4);
  Matrix<double,Dynamic,1> v3(4);
  EXPECT_EQ(4U, size_of(v1));
  EXPECT_TRUE(check_consistent_sizes(function,v1,v2,v3,name1,name2,name3,&result));
}

