#include <stan/math/error_handling/matrix/check_unit_vector.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkUnitVector) {
  Eigen::Matrix<double,Eigen::Dynamic,1> y(2);
  double result;
  y << sqrt(0.5), sqrt(0.5);
  
  EXPECT_TRUE(stan::math::check_unit_vector("checkUnitVector(%1%)",
                                        y, "y", &result));

  y[1] = 0.55;
  EXPECT_THROW(stan::math::check_unit_vector("checkUnitVector(%1%)", y, "y", &result), 
               std::domain_error);
}

TEST(MathErrorHandlingMatrix, checkUnitVector_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,1> y(2);
  double result;
  double nan = std::numeric_limits<double>::quiet_NaN();

  y << nan, sqrt(0.5);
  EXPECT_THROW(stan::math::check_unit_vector("checkUnitVector(%1%)", y, "y", &result), 
               std::domain_error);
  y << sqrt(0.5), nan;
  EXPECT_THROW(stan::math::check_unit_vector("checkUnitVector(%1%)", y, "y", &result), 
               std::domain_error);
  y << nan, nan;
  EXPECT_THROW(stan::math::check_unit_vector("checkUnitVector(%1%)", y, "y", &result), 
               std::domain_error);
}
