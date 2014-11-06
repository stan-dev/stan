#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingScalar, checkConsistentSizes) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::error_handling::check_consistent_sizes;
  using stan::size_of;

  const std::string function = "testConsSizes";
  const std::string name1 = "name1";
  const std::string name2 = "name2";
  const std::string name3 = "name3";
  const std::string name4 = "name4";
  

  Matrix<double,Dynamic,1> v1(4);
  Matrix<double,Dynamic,1> v2(4);
  Matrix<double,Dynamic,1> v3(4);
  Matrix<double,Dynamic,1> v4(4);
  ASSERT_EQ(4U, size_of(v1));
  ASSERT_EQ(4U, size_of(v2));
  ASSERT_EQ(4U, size_of(v3));
  ASSERT_EQ(4U, size_of(v4));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v2));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v2, name3, v3));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v2, name3, v3, name4, v4));
  
  Matrix<double,Dynamic,1> v(3);
  
  ASSERT_EQ(3U, size_of(v));
  const std::string name = "inconsistent";
  EXPECT_THROW(check_consistent_sizes(function, name, v, name2, v2),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name, v),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function, name, v, name2, v2, name3, v3),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name, v, name3, v3),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name2, v2, name, v),
               std::domain_error);

  EXPECT_THROW(check_consistent_sizes(function, name, v, name2, v2, name3, v3, name4, v4),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name, v, name3, v3, name4, v4),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name2, v2, name, v, name4, v4),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function, name, v, name2, v2, name3, v3, name, v),
               std::domain_error);
}

TEST(ErrorHandlingScalar, checkConsistentSizes_nan) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::error_handling::check_consistent_sizes;
  using stan::size_of;

  const std::string function = "testConsSizes";
  const std::string name1 = "name1";
  const std::string name2 = "name2";
  const std::string name3 = "name3";
  const std::string name4 = "name4";
  
  double nan = std::numeric_limits<double>::quiet_NaN();

  Matrix<double,Dynamic,1> v1(4);
  Matrix<double,Dynamic,1> v2(4);
  Matrix<double,Dynamic,1> v3(4);
  Matrix<double,Dynamic,1> v4(4);
  v1 << nan,1,2,nan;
  v2 << nan,1,2,nan;
  v3 << nan,1,2,nan;
  v4 << nan,1,2,nan;

  ASSERT_EQ(4U, size_of(v1));
  ASSERT_EQ(4U, size_of(v2));
  ASSERT_EQ(4U, size_of(v3));
  ASSERT_EQ(4U, size_of(v4));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v2));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v2, name3, v3));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v2, name3, v3, name4, v4));
  
  Matrix<double,Dynamic,1> v(3);
  v << nan,1,2;
  ASSERT_EQ(3U, size_of(v));
  const std::string name = "inconsistent";
  EXPECT_THROW(check_consistent_sizes(function, name, v, name2, v2),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name, v),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function, name, v, name2, v2, name3, v3),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name, v, name3, v3),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name2, v2, name, v),
               std::domain_error);

  EXPECT_THROW(check_consistent_sizes(function, name, v, name2, v2, name3, v3, name4, v4),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name, v, name3, v3, name4, v4),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function, name1, v1, name2, v2, name, v, name4, v4),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function, name, v, name2, v2, name3, v3, name, v),
               std::domain_error);
}

