#include <stan/math/error_handling/matrix/check_cholesky_factor.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkCovMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double result;

  y.resize(1,1);
  y << 1;
  EXPECT_TRUE(stan::math::check_cholesky_factor("checkCovMatrix(%1%)",
                                           y, "y", &result));
  
  y.resize(3,3);
  y << 
    1, 0, 0,
    1, 1, 0,
    1, 1, 1;
  EXPECT_TRUE(stan::math::check_cholesky_factor("checkCovMatrix(%1%)",
                                           y, "y", &result));

  y.resize(1,1);
  y << -1;
  EXPECT_THROW(stan::math::check_cholesky_factor("checkCovMatrix(%1%)", 
                                              y, "y", &result), 
               std::domain_error);

  y.resize(3,3);
  y << 
    1, 2, 3, 
    4, 5, 6, 
    7, 8, 9;
  EXPECT_THROW(stan::math::check_cholesky_factor("checkCovMatrix(%1%)", 
                                                 y, "y", &result), 
               std::domain_error);

  y.resize(3,3);
  y <<
    1, 0, 0, 
    2, -1, 0,
    1, 2, 3;
  EXPECT_THROW(stan::math::check_cholesky_factor("checkCovMatrix(%1%)", 
                                                 y, "y", &result), 
               std::domain_error);


  y.resize(2,3);
  y << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(stan::math::check_cholesky_factor("checkCovMatrix(%1%)", 
                                                 y, "y", &result),
               std::domain_error);

  y.resize(3,2);
  y << 
    1, 0,
    2, 3,
    4, 5;
  EXPECT_TRUE(stan::math::check_cholesky_factor("checkCovMatrix(%1%)",
                                                y, "y", &result));
  y(0,1) = 1.5;
  EXPECT_THROW(stan::math::check_cholesky_factor("checkCovMatrix(%1%)", 
                                                 y, "y", &result),
               std::domain_error);

  

}


