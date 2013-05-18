#include <stan/agrad/fwd/matrix/dot_self.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/math/matrix.hpp>

TEST(AgradFwdMatrix,columns_dot_self) {
  using stan::math::columns_dot_self;
  using stan::agrad::fvar;

  Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  m1(0).d_ = 1.0;
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).val_,1E-12);
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).d_,1E-12);

  Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic> m2(1,2);
  m2 << 2.0, 3.0;
  m2(0).d_ = 1.0;
  m2(1).d_ = 1.0;
  Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic> x;
  x = columns_dot_self(m2);
  EXPECT_NEAR(4.0,x(0,0).val_,1E-12);
  EXPECT_NEAR(9.0,x(0,1).val_,1E-12);
  EXPECT_NEAR(4.0,x(0,0).d_,1E-12);
  EXPECT_NEAR(6.0,x(0,1).d_,1E-12);

  Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  m3(0,0).d_ = 1.0;
  m3(0,1).d_ = 1.0;
  m3(1,0).d_ = 1.0;
  m3(1,1).d_ = 1.0;
  x = columns_dot_self(m3);
  EXPECT_NEAR(20.0,x(0,0).val_,1E-12);
  EXPECT_NEAR(34.0,x(0,1).val_,1E-12);
  EXPECT_NEAR(12.0,x(0,0).d_,1E-12);
  EXPECT_NEAR(16.0,x(0,1).d_,1E-12);
}
