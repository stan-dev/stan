#include <stan/error_handling/scalar/check_consistent_size.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingScalar, checkConsistentSize) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::error_handling::check_consistent_size;
  using stan::size_of;

  const std::string function = "checkConsistentSize";
  const std::string name1 = "name1";
  

  Matrix<double,Dynamic,1> v1(4);
  EXPECT_EQ(4U, size_of(v1));
  EXPECT_TRUE(check_consistent_size(function, name1, v1, 4U));
  EXPECT_THROW(check_consistent_size(function, name1, v1, 2U), std::domain_error);
}

TEST(ErrorHandlingScalar, checkConsistentSize_nan) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::error_handling::check_consistent_size;
  using stan::size_of;

  const std::string function = "checkConsistentSize";
  const std::string name1 = "name1";
  
  double nan = std::numeric_limits<double>::quiet_NaN();

  Matrix<double,Dynamic,1> v1(4);
  v1 << nan,nan,4,nan;
  EXPECT_EQ(4U, size_of(v1));
  EXPECT_TRUE(check_consistent_size(function, name1, v1, 4U));
  EXPECT_THROW(check_consistent_size(function, name1, v1, 2U), std::domain_error);
}
