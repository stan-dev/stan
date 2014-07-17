#include <stan/math/error_handling/matrix/check_simplex.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkSimplex) {
  Eigen::Matrix<double,Eigen::Dynamic,1> y(2);
  double result;
  y << 0.5, 0.5;
  
  EXPECT_TRUE(stan::math::check_simplex("checkSimplex(%1%)",
                                        y, "y", &result));
                  
  y[1] = 0.55;
  EXPECT_THROW(stan::math::check_simplex("checkSimplex(%1%)", 
                                         y, "y", &result), 
               std::domain_error);
}
