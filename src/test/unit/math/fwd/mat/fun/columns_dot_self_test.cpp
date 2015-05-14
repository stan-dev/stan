#include <stan/math/fwd/mat/fun/dot_self.hpp>
#include <stan/math/fwd/mat/fun/columns_dot_self.hpp>
#include <stan/math/prim/mat/fun/columns_dot_self.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdMatrixColumnsDotSelf,matrix_fd) {
  using stan::math::columns_dot_self;
  using stan::math::matrix_fd;

  matrix_fd m1(1,1);
  m1 << 2.0;
  m1(0).d_ = 1.0;
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).val_,1E-12);
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).d_,1E-12);

  matrix_fd m2(1,2);
  m2 << 2.0, 3.0;
  m2(0).d_ = 1.0;
  m2(1).d_ = 1.0;
  matrix_fd x;
  x = columns_dot_self(m2);
  EXPECT_NEAR(4.0,x(0,0).val_,1E-12);
  EXPECT_NEAR(9.0,x(0,1).val_,1E-12);
  EXPECT_NEAR(4.0,x(0,0).d_,1E-12);
  EXPECT_NEAR(6.0,x(0,1).d_,1E-12);

  matrix_fd m3(2,2);
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
TEST(AgradFwdMatrixColumnsDotSelf,matrix_ffd) {
  using stan::math::columns_dot_self;
  using stan::math::matrix_ffd;
  using stan::math::fvar;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 4.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 5.0;
  d.d_.val_ = 1.0;  

  matrix_ffd m1(1,1);
  m1 << a;

  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).val_.val(),1E-12);
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).d_.val(),1E-12);

  matrix_ffd m2(1,2);
  m2 << a,b;
  matrix_ffd x;
  x = columns_dot_self(m2);
  EXPECT_NEAR(4.0,x(0,0).val_.val(),1E-12);
  EXPECT_NEAR(9.0,x(0,1).val_.val(),1E-12);
  EXPECT_NEAR(4.0,x(0,0).d_.val(),1E-12);
  EXPECT_NEAR(6.0,x(0,1).d_.val(),1E-12);

  matrix_ffd m3(2,2);
  m3 << a,b,c,d;
  x = columns_dot_self(m3);
  EXPECT_NEAR(20.0,x(0,0).val_.val(),1E-12);
  EXPECT_NEAR(34.0,x(0,1).val_.val(),1E-12);
  EXPECT_NEAR(12.0,x(0,0).d_.val(),1E-12);
  EXPECT_NEAR(16.0,x(0,1).d_.val(),1E-12);
}
