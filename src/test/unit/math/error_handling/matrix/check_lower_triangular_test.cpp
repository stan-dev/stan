#include <stan/math/error_handling/matrix/check_lower_triangular.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkLowerTriangular) {
  using stan::math::check_lower_triangular;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;
  
  y.resize(1,1);
  y << 1;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular(%1%)", y, "y",
                                     &result));

  y.resize(1,2);
  y << 1, 0;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular(%1%)", y, "y", 
                                     &result));

  y(0,1) = 1;
  EXPECT_THROW(check_lower_triangular("checkLowerTriangular(%1%)", y, "y", 
                                      &result), 
               std::domain_error);
  
  

  y.resize(2,2);
  y << 1, 0, 2, 3;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular(%1%)", y, "y", 
                                     &result));

  y << 1, 2, 3, 4;
  EXPECT_THROW(check_lower_triangular("checkLowerTriangular(%1%)", y, "y", 
                                      &result), 
               std::domain_error);

  y.resize(3,2);
  y << 1, 0,
    2, 3,
    4, 5;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular(%1%)", y, "y", 
                                     &result));
  
  y(0,1) = 1.5;
  EXPECT_THROW(check_lower_triangular("checkLowerTriangular(%1%)", y, "y", 
                                      &result),
               std::domain_error);
  
  y.resize(2,3);
  y << 
    1, 0, 0,
    4, 5, 0;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular(%1%)", y, "y", 
                                     &result));
  y(0,2) = 3;
}
