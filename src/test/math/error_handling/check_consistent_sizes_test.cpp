#include <stan/math/error_handling/check_consistent_sizes.hpp>
#include <gtest/gtest.h>

TEST(MathMatrixErrorHandling, checkConsistentSizes) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::check_consistent_sizes;
  using stan::size_of;

  const char* function = "testConsSizes(%1%)";
  const char* name1 = "name1";
  const char* name2 = "name2";
  const char* name3 = "name3";
  const char* name4 = "name4";
  
  double result;

  Matrix<double,Dynamic,1> v1(4);
  Matrix<double,Dynamic,1> v2(4);
  Matrix<double,Dynamic,1> v3(4);
  Matrix<double,Dynamic,1> v4(4);
  ASSERT_EQ(4U, size_of(v1));
  ASSERT_EQ(4U, size_of(v2));
  ASSERT_EQ(4U, size_of(v3));
  ASSERT_EQ(4U, size_of(v4));
  EXPECT_TRUE(check_consistent_sizes(function,v1,v2,name1,name2,&result));
  EXPECT_TRUE(check_consistent_sizes(function,v1,v2,v3,name1,name2,name3,&result));
  EXPECT_TRUE(check_consistent_sizes(function,v1,v2,v3,v4,name1,name2,name3,name4,
                                     &result));
  
  Matrix<double,Dynamic,1> v(3);
  
  ASSERT_EQ(3U, size_of(v));
  const char* name = "inconsistent";
  EXPECT_THROW(check_consistent_sizes(function,v,v2,name,name2,&result),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function,v1,v,name1,name,&result),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function,v,v2,v3,name,name2,name3,&result),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function,v1,v,v3,name1,name,name3,&result),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function,v1,v2,v,name1,name2,name,&result),
               std::domain_error);

  EXPECT_THROW(check_consistent_sizes(function,v,v2,v3,v4,
                                      name,name2,name3,name4,
                                      &result),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function,v1,v,v3,v4,
                                      name1,name,name3,name4,
                                      &result),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function,v1,v2,v,v4,
                                      name1,name2,name,name4,
                                      &result),
               std::domain_error);
  EXPECT_THROW(check_consistent_sizes(function,v1,v2,v3,v,
                                      name,name2,name3,name,
                                      &result),
               std::domain_error);
}

