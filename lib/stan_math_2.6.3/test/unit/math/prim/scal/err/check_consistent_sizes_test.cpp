#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingScalar, checkConsistentSizes_zero) {
  using stan::math::check_consistent_sizes;
  const char* function = "testConsSizes";
  const char* name1 = "name1";
  const char* name2 = "name2";
  const char* name3 = "name3";
  const char* name4 = "name4";

  std::vector<double> v1(0);
  double d1;

  ASSERT_EQ(0, stan::size_of(v1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, d1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, d1, name2, v1));

  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v1, name3, v1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v1, name3, d1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, d1, name3, v1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, d1, name3, d1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, d1, name2, v1, name3, v1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, d1, name2, v1, name3, d1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, d1, name2, d1, name3, v1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, d1, name2, d1, name3, d1));

  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v1, name3, v1, name4, v1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v1, name3, v1, name4, d1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v1, name3, d1, name4, v1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, v1, name3, d1, name4, d1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, d1, name3, v1, name4, v1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, d1, name3, v1, name4, d1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, d1, name3, d1, name4, v1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, v1, name2, d1, name3, d1, name4, d1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, d1, name2, v1, name3, v1, name4, v1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, d1, name2, v1, name3, v1, name4, d1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, d1, name2, v1, name3, d1, name4, v1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, d1, name2, v1, name3, d1, name4, d1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, d1, name2, d1, name3, v1, name4, v1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, d1, name2, d1, name3, v1, name4, d1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, d1, name2, d1, name3, d1, name4, v1));
  EXPECT_TRUE(check_consistent_sizes(function, name1, d1, name2, d1, name3, d1, name4, d1));
}

TEST(ErrorHandlingScalar, checkConsistentSizes) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::check_consistent_sizes;
  using stan::size_of;

  const char* function = "testConsSizes";
  const char* name1 = "name1";
  const char* name2 = "name2";
  const char* name3 = "name3";
  const char* name4 = "name4";
  

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
}

TEST(ErrorHandlingScalar, checkConsistentSizes_nan) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::check_consistent_sizes;
  using stan::size_of;

  const char* function = "testConsSizes";
  const char* name1 = "name1";
  const char* name2 = "name2";
  const char* name3 = "name3";
  const char* name4 = "name4";
  
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
}

