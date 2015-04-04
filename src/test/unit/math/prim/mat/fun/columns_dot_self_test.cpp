#include <stan/math/prim/mat/fun/columns_dot_self.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,columns_dot_self) {
  using stan::math::columns_dot_self;

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m2(1,2);
  m2 << 2.0, 3.0;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  x = columns_dot_self(m2);
  EXPECT_NEAR(4.0,x(0,0),1E-12);
  EXPECT_NEAR(9.0,x(0,1),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  x = columns_dot_self(m3);
  EXPECT_NEAR(20.0,x(0,0),1E-12);
  EXPECT_NEAR(34.0,x(0,1),1E-12);
}
