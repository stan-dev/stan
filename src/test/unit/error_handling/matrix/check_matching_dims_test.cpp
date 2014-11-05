#include <stan/error_handling/matrix/check_matching_dims.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingMatrix, checkMatchingDimsMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  
  y.resize(3,3);
  x.resize(3,3);
  EXPECT_TRUE(stan::error_handling::check_matching_dims("checkMatchingDims", "x", x,
                                                        "y", y));
  x.resize(0,0);
  y.resize(0,0);
  EXPECT_TRUE(stan::error_handling::check_matching_dims("checkMatchingDims", "x", x,
                                                        "y", y));

  y.resize(1,2);
  EXPECT_THROW(stan::error_handling::check_matching_dims("checkMatchingDims", "x", x,
                                                         "y", y), 
               std::domain_error);

  x.resize(2,1);
  EXPECT_THROW(stan::error_handling::check_matching_dims("checkMatchingDims", "x", x,
                                                         "y", y), 
               std::domain_error);
}

TEST(ErrorHandlingMatrix, checkMatchingDimsMatrix_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  double nan = std::numeric_limits<double>::quiet_NaN();

  y.resize(3,3);
  x.resize(3,3);
  y << nan, nan, nan,nan, nan, nan,nan, nan, nan;
  x << nan, nan, nan,nan, nan, nan,nan, nan, nan;
  EXPECT_TRUE(stan::error_handling::check_matching_dims("checkMatchingDims", "x", x,
                                                        "y", y));
  x.resize(0,0);
  y.resize(0,0);
  EXPECT_TRUE(stan::error_handling::check_matching_dims("checkMatchingDims", "x", x,
                                                        "y", y));

  y.resize(1,2);
  y << nan, nan;
  EXPECT_THROW(stan::error_handling::check_matching_dims("checkMatchingDims", "x", x,
                                                         "y", y), 
               std::domain_error);

  x.resize(2,1);
  x << nan, nan;
  EXPECT_THROW(stan::error_handling::check_matching_dims("checkMatchingDims", "x", x,
                                                         "y", y), 
               std::domain_error);
}
