#include <stan/agrad/fwd/matrix/dot_self.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/math/matrix.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixColumnsDotSelf,matrix_fd) {
  using stan::math::columns_dot_self;
  using stan::agrad::matrix_fd;

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
TEST(AgradFwdMatrixColumnsDotSelf,matrix_fv_1stDeriv) {
  using stan::math::columns_dot_self;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c(4.0,1.0);
  fvar<var> d(5.0,1.0);
  matrix_fv m1(1,1);
  m1 << a;

  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).val_.val(),1E-12);
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).d_.val(),1E-12);

  matrix_fv m2(1,2);
  m2 << a,b;
  matrix_fv x;
  x = columns_dot_self(m2);
  EXPECT_NEAR(4.0,x(0,0).val_.val(),1E-12);
  EXPECT_NEAR(9.0,x(0,1).val_.val(),1E-12);
  EXPECT_NEAR(4.0,x(0,0).d_.val(),1E-12);
  EXPECT_NEAR(6.0,x(0,1).d_.val(),1E-12);

  matrix_fv m3(2,2);
  m3 << a,b,c,d;
  x = columns_dot_self(m3);
  EXPECT_NEAR(20.0,x(0,0).val_.val(),1E-12);
  EXPECT_NEAR(34.0,x(0,1).val_.val(),1E-12);
  EXPECT_NEAR(12.0,x(0,0).d_.val(),1E-12);
  EXPECT_NEAR(16.0,x(0,1).d_.val(),1E-12);

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  x(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(8.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixColumnsDotSelf,matrix_fv_2ndDeriv) {
  using stan::math::columns_dot_self;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(2.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c(4.0,1.0);
  fvar<var> d(5.0,1.0);
  matrix_fv x;
  matrix_fv m3(2,2);
  m3 << a,b,c,d;
  x = columns_dot_self(m3);

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  x(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(2.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixColumnsDotSelf,matrix_ffd) {
  using stan::math::columns_dot_self;
  using stan::agrad::matrix_ffd;
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
TEST(AgradFwdMatrixColumnsDotSelf,matrix_ffv_1stDeriv) {
  using stan::math::columns_dot_self;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(2.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(4.0,1.0);
  fvar<fvar<var> > d(5.0,1.0);
  matrix_ffv m1(1,1);
  m1 << a;

  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).val_.val().val(),1E-12);
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).d_.val().val(),1E-12);

  matrix_ffv m2(1,2);
  m2 << a,b;
  matrix_ffv x;
  x = columns_dot_self(m2);
  EXPECT_NEAR(4.0,x(0,0).val_.val().val(),1E-12);
  EXPECT_NEAR(9.0,x(0,1).val_.val().val(),1E-12);
  EXPECT_NEAR(4.0,x(0,0).d_.val().val(),1E-12);
  EXPECT_NEAR(6.0,x(0,1).d_.val().val(),1E-12);

  matrix_ffv m3(2,2);
  m3 << a,b,c,d;
  x = columns_dot_self(m3);
  EXPECT_NEAR(20.0,x(0,0).val_.val().val(),1E-12);
  EXPECT_NEAR(34.0,x(0,1).val_.val().val(),1E-12);
  EXPECT_NEAR(12.0,x(0,0).d_.val().val(),1E-12);
  EXPECT_NEAR(16.0,x(0,1).d_.val().val(),1E-12);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  x(0,0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(8.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixColumnsDotSelf,matrix_ffv_2ndDeriv_1) {
  using stan::math::columns_dot_self;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(2.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(4.0,1.0);
  fvar<fvar<var> > d(5.0,1.0);
  matrix_ffv x;
  matrix_ffv m3(2,2);
  m3 << a,b,c,d;
  x = columns_dot_self(m3);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  x(0,0).val().d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}

TEST(AgradFwdMatrixColumnsDotSelf,matrix_ffv_2ndDeriv_2) {
  using stan::math::columns_dot_self;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(2.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(4.0,1.0);
  fvar<fvar<var> > d(5.0,1.0);
  matrix_ffv x;
  matrix_ffv m3(2,2);
  m3 << a,b,c,d;
  x = columns_dot_self(m3);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  x(0,0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(2.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradFwdMatrixColumnsDotSelf,matrix_ffv_3rdDeriv) {
  using stan::math::columns_dot_self;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(2.0,1.0);
  fvar<fvar<var> > b(3.0,1.0);
  fvar<fvar<var> > c(4.0,1.0);
  fvar<fvar<var> > d(5.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  matrix_ffv x;
  matrix_ffv m3(2,2);
  m3 << a,b,c,d;
  x = columns_dot_self(m3);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  x(0,0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
