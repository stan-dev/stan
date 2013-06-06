#include <stan/agrad/fwd/matrix/dot_self.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/math/matrix.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFwdMatrix,columns_dot_self) {
  using stan::math::columns_dot_self;
  using stan::agrad::matrix_fv;

  matrix_fv m1(1,1);
  m1 << 2.0;
  m1(0).d_ = 1.0;
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).val_,1E-12);
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).d_,1E-12);

  matrix_fv m2(1,2);
  m2 << 2.0, 3.0;
  m2(0).d_ = 1.0;
  m2(1).d_ = 1.0;
  matrix_fv x;
  x = columns_dot_self(m2);
  EXPECT_NEAR(4.0,x(0,0).val_,1E-12);
  EXPECT_NEAR(9.0,x(0,1).val_,1E-12);
  EXPECT_NEAR(4.0,x(0,0).d_,1E-12);
  EXPECT_NEAR(6.0,x(0,1).d_,1E-12);

  matrix_fv m3(2,2);
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
TEST(AgradFwdFvarVarMatrix,columns_dot_self) {
  using stan::math::columns_dot_self;
  using stan::agrad::matrix_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c(4.0,1.0);
  fvar<var> d(5.0,1.0);
  matrix_fvv m1(1,1);
  m1 << a;

  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).val_.val(),1E-12);
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).d_.val(),1E-12);

  matrix_fvv m2(1,2);
  m2 << a,b;
  matrix_fvv x;
  x = columns_dot_self(m2);
  EXPECT_NEAR(4.0,x(0,0).val_.val(),1E-12);
  EXPECT_NEAR(9.0,x(0,1).val_.val(),1E-12);
  EXPECT_NEAR(4.0,x(0,0).d_.val(),1E-12);
  EXPECT_NEAR(6.0,x(0,1).d_.val(),1E-12);

  matrix_fvv m3(2,2);
  m3 << a,b,c,d;
  x = columns_dot_self(m3);
  EXPECT_NEAR(20.0,x(0,0).val_.val(),1E-12);
  EXPECT_NEAR(34.0,x(0,1).val_.val(),1E-12);
  EXPECT_NEAR(12.0,x(0,0).d_.val(),1E-12);
  EXPECT_NEAR(16.0,x(0,1).d_.val(),1E-12);
}
TEST(AgradFwdFvarFvarMatrix,columns_dot_self) {
  using stan::math::columns_dot_self;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;

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

  matrix_ffv m1(1,1);
  m1 << a;

  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).val_.val(),1E-12);
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).d_.val(),1E-12);

  matrix_ffv m2(1,2);
  m2 << a,b;
  matrix_ffv x;
  x = columns_dot_self(m2);
  EXPECT_NEAR(4.0,x(0,0).val_.val(),1E-12);
  EXPECT_NEAR(9.0,x(0,1).val_.val(),1E-12);
  EXPECT_NEAR(4.0,x(0,0).d_.val(),1E-12);
  EXPECT_NEAR(6.0,x(0,1).d_.val(),1E-12);

  matrix_ffv m3(2,2);
  m3 << a,b,c,d;
  x = columns_dot_self(m3);
  EXPECT_NEAR(20.0,x(0,0).val_.val(),1E-12);
  EXPECT_NEAR(34.0,x(0,1).val_.val(),1E-12);
  EXPECT_NEAR(12.0,x(0,0).d_.val(),1E-12);
  EXPECT_NEAR(16.0,x(0,1).d_.val(),1E-12);
}
