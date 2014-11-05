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
               std::domain_error);

  x.resize(2,2);
  EXPECT_THROW(stan::error_handling::check_multiplicable("checkMultiplicable", "x", x,
                                                         "y", y), 
               std::domain_error);
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
               std::domain_error);

  x.resize(2,2);
  x << nan, nan, nan, nan;
  EXPECT_THROW(stan::error_handling::check_multiplicable("checkMultiplicable", "x", x,
                                                         "y", y), 
               std::domain_error);
}
