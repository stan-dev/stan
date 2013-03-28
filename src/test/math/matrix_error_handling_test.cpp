#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/error_handling/check_not_nan.hpp>
#include <stan/math/error_handling/check_consistent_size.hpp>
#include <stan/math/error_handling/check_consistent_sizes.hpp>
#include <gtest/gtest.h>


using stan::math::default_policy;



TEST(MathMatrixErrorHandling, checkConsistentSize) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::check_consistent_size;
  using stan::size_of;

  const char* function = "checkConsistentSize(%1%)";
  const char* name1 = "name1";
  
  double result;

  Matrix<double,Dynamic,1> v1(4);
  EXPECT_EQ(4U, size_of(v1));
  EXPECT_TRUE(check_consistent_size(4U, function, v1, name1, &result, default_policy()));
}

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

