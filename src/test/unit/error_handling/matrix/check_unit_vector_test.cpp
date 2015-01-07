#include <stan/error_handling/matrix/check_unit_vector.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingMatrix, checkUnitVector) {
  Eigen::Matrix<double,Eigen::Dynamic,1> y(2);
  y << sqrt(0.5), sqrt(0.5);
  
  EXPECT_TRUE(stan::error_handling::check_unit_vector("checkUnitVector",
                                                      "y", y));

  y[1] = 0.55;
  EXPECT_THROW(stan::error_handling::check_unit_vector("checkUnitVector", "y", y),
               std::domain_error);
}

TEST(ErrorHandlingMatrix, checkUnitVector_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,1> y(2);
  double nan = std::numeric_limits<double>::quiet_NaN();

  y << nan, sqrt(0.5);
  EXPECT_THROW(stan::error_handling::check_unit_vector("checkUnitVector", "y", y),
               std::domain_error);
  y << sqrt(0.5), nan;
  EXPECT_THROW(stan::error_handling::check_unit_vector("checkUnitVector", "y", y),
               std::domain_error);
  y << nan, nan;
  EXPECT_THROW(stan::error_handling::check_unit_vector("checkUnitVector", "y", y),
               std::domain_error);
}
