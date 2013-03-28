#include <stan/math/error_handling/matrix/check_unit_vector.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkUnitVector) {
  Eigen::Matrix<double,Eigen::Dynamic,1> y(2);
  double result;
  y << sqrt(0.5), sqrt(0.5);
  
  EXPECT_TRUE(stan::math::check_unit_vector("checkUnitVector(%1%)",
                                        y, "y", &result));
  EXPECT_TRUE(stan::math::check_unit_vector("checkUnitVector(%1%)",
                                        y, "y"));
                  
  y[1] = 0.55;
  EXPECT_THROW(stan::math::check_unit_vector("checkUnitVector(%1%)", y, "y", &result), 
               std::domain_error);
  EXPECT_THROW(stan::math::check_unit_vector("checkUnitVector(%1%)", y, "y"),
               std::domain_error);
}
