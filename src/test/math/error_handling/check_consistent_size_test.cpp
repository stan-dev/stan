#include <stan/math/error_handling/check_consistent_size.hpp>
#include <gtest/gtest.h>
#include <stan/math/error_handling/default_policy.hpp>

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
