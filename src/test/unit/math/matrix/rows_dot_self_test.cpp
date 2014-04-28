#include <stan/math/matrix/rows_dot_self.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,rows_dot_self) {
  using stan::math::rows_dot_self;

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  EXPECT_NEAR(4.0,rows_dot_self(m1)(0,0),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m2(1,2);
  m2 << 2.0, 3.0;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x,y;
  x = rows_dot_self(m2);
  EXPECT_NEAR(13.0,x(0,0),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  y = rows_dot_self(m3);
  EXPECT_NEAR(13.0,y(0,0),1E-12);
  EXPECT_NEAR(41.0,y(1,0),1E-12);
}
