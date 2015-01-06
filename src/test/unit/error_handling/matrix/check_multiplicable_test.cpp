#include <iostream>
#include <stan/error_handling/matrix/check_multiplicable.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingMatrix, checkMultiplicableMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  
  y.resize(3,3);
  x.resize(3,3);
  EXPECT_TRUE(stan::error_handling::check_multiplicable("checkMultiplicable", "x", x,
                                                        "y", y));
  x.resize(3,2);
  y.resize(2,4);
  EXPECT_TRUE(stan::error_handling::check_multiplicable("checkMultiplicable", "x", x,
                                                        "y", y));

  y.resize(1,2);
  EXPECT_THROW(stan::error_handling::check_multiplicable("checkMultiplicable", "x", x,
                                                         "y", y), 
               std::invalid_argument);

  x.resize(2,2);
  EXPECT_THROW(stan::error_handling::check_multiplicable("checkMultiplicable", "x", x,
                                                         "y", y), 
               std::invalid_argument);
}

TEST(ErrorHandlingMatrix, checkMultiplicableMatrix_0) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  
  x.resize(3,0);
  y.resize(0,3);
  EXPECT_THROW(stan::error_handling::check_multiplicable("checkMultiplicable", "x", x,
                                                         "y", y),
               std::invalid_argument);

  x.resize(0,4);
  y.resize(4,3);
  EXPECT_THROW(stan::error_handling::check_multiplicable("checkMultiplicable", "x", x,
                                                         "y", y),
               std::invalid_argument);

  x.resize(3,4);
  y.resize(4,0);
  EXPECT_THROW(stan::error_handling::check_multiplicable("checkMultiplicable", "x", x,
                                                         "y", y),
               std::invalid_argument);
}

TEST(ErrorHandlingMatrix, checkMultiplicableMatrix_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  double nan = std::numeric_limits<double>::quiet_NaN();
    
  y.resize(3,3);
  x.resize(3,3);
  y << nan, nan, nan,nan, nan, nan,nan, nan, nan;
  x << nan, nan, nan,nan, nan, nan,nan, nan, nan;
  EXPECT_TRUE(stan::error_handling::check_multiplicable("checkMultiplicable", "x", x,
                                                        "y", y));
  x.resize(3,2);
  y.resize(2,4);
  y << nan, nan, nan,nan, nan, nan,nan, nan;
  x << nan, nan, nan,nan, nan, nan;
  EXPECT_TRUE(stan::error_handling::check_multiplicable("checkMultiplicable", "x", x,
                                                        "y", y));

  y.resize(1,2);
  y << nan, nan;
  EXPECT_THROW(stan::error_handling::check_multiplicable("checkMultiplicable", "x", x,
                                                         "y", y), 
               std::invalid_argument);

  x.resize(2,2);
  x << nan, nan, nan, nan;
  EXPECT_THROW(stan::error_handling::check_multiplicable("checkMultiplicable", "x", x,
                                                         "y", y), 
               std::invalid_argument);
}
