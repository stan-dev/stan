#include <stan/agrad/rev/matrix/multiply.hpp>
#include <stan/agrad/rev/operators.hpp>
#include <stan/agrad/rev/functions/abs.hpp>
#include <stan/agrad/fwd/matrix/inverse.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradFwdMatrixInverse,fd) {
  using stan::agrad::matrix_fd;
  using stan::math::matrix_d;

  matrix_fd a(2,2);
  a << 2.0, 3.0, 5.0, 7.0;
   a(0,0).d_ = 1.0;
   a(0,1).d_ = 1.0;
   a(1,0).d_ = 1.0;
   a(1,1).d_ = 1.0;

   matrix_d b(2,2);
   b << 2.0, 3.0, 5.0,7.0;
   b = b.inverse();

  matrix_fd a_inv = stan::agrad::inverse(a);

  EXPECT_NEAR(b(0,0),a_inv(0,0).val_,1.0E-12);
  EXPECT_NEAR(b(0,1),a_inv(0,1).val_,1.0E-12);
  EXPECT_NEAR(b(1,0),a_inv(1,0).val_,1.0E-12);
  EXPECT_NEAR(b(1,1),a_inv(1,1).val_,1.0E-12);
  EXPECT_NEAR(-8,a_inv(0,0).d_,1.0E-12);
  EXPECT_NEAR( 4,a_inv(0,1).d_,1.0E-12);
  EXPECT_NEAR( 6,a_inv(1,0).d_,1.0E-12);
  EXPECT_NEAR(-3,a_inv(1,1).d_,1.0E-12);

  EXPECT_THROW(stan::agrad::inverse(matrix_fd(2,3)), std::domain_error);
}
TEST(AgradFwdMatrixInverse,fv_1stDeriv) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> d(2.0,1.0);
  fvar<var> e(3.0,1.0);
  fvar<var> f(5.0,1.0);
  fvar<var> g(7.0,1.0);

  matrix_fv a(2,2);
  a << d,e,f,g;

   matrix_d b(2,2);
   b << 2.0, 3.0, 5.0,7.0;
   b = b.inverse();

  matrix_fv a_inv = stan::agrad::inverse(a);

  EXPECT_NEAR(b(0,0),a_inv(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(b(0,1),a_inv(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(b(1,0),a_inv(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(b(1,1),a_inv(1,1).val_.val(),1.0E-12);
  EXPECT_NEAR(-8,a_inv(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR( 4,a_inv(0,1).d_.val(),1.0E-12);
  EXPECT_NEAR( 6,a_inv(1,0).d_.val(),1.0E-12);
  EXPECT_NEAR(-3,a_inv(1,1).d_.val(),1.0E-12);

  EXPECT_THROW(stan::agrad::inverse(matrix_fv(2,3)), std::domain_error);

  AVEC q = createAVEC(d.val(),e.val(),f.val(),g.val());
  VEC h;
  a_inv(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-49.0,h[0]);
  EXPECT_FLOAT_EQ(35.0,h[1]);
  EXPECT_FLOAT_EQ(21.0,h[2]);
  EXPECT_FLOAT_EQ(-15.0,h[3]);
}
TEST(AgradFwdMatrixInverse,fv_2ndDeriv) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> d(2.0,1.0);
  fvar<var> e(3.0,1.0);
  fvar<var> f(5.0,1.0);
  fvar<var> g(7.0,1.0);

  matrix_fv a(2,2);
  a << d,e,f,g;

  matrix_fv a_inv = stan::agrad::inverse(a);

  AVEC q = createAVEC(d.val(),e.val(),f.val(),g.val());
  VEC h;
  a_inv(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-112.0,h[0]);
  EXPECT_FLOAT_EQ(82.0,h[1]);
  EXPECT_FLOAT_EQ(52.0,h[2]);
  EXPECT_FLOAT_EQ(-38.0,h[3]);
}
TEST(AgradFwdMatrixInverse,ffd) {
  using stan::agrad::matrix_ffd;
  using stan::math::matrix_d;
  using stan::agrad::fvar;

  fvar<fvar<double> > d,e,f,g;
  d.val_.val_ = 2.0;
  d.d_.val_ = 1.0;  
  e.val_.val_ = 3.0;
  e.d_.val_ = 1.0;
  f.val_.val_ = 5.0;
  f.d_.val_ = 1.0;
  g.val_.val_ = 7.0;
  g.d_.val_ = 1.0;  

  matrix_ffd a(2,2);
  a << d,e,f,g;

   matrix_d b(2,2);
   b << 2.0, 3.0, 5.0,7.0;
   b = b.inverse();

  matrix_ffd a_inv = stan::agrad::inverse(a);

  EXPECT_NEAR(b(0,0),a_inv(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(b(0,1),a_inv(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(b(1,0),a_inv(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(b(1,1),a_inv(1,1).val_.val(),1.0E-12);
  EXPECT_NEAR(-8,a_inv(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR( 4,a_inv(0,1).d_.val(),1.0E-12);
  EXPECT_NEAR( 6,a_inv(1,0).d_.val(),1.0E-12);
  EXPECT_NEAR(-3,a_inv(1,1).d_.val(),1.0E-12);

  EXPECT_THROW(stan::agrad::inverse(matrix_ffd(2,3)), std::domain_error);
}
TEST(AgradFwdMatrixInverse,ffv_1stDeriv) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > d(2.0,1.0);
  fvar<fvar<var> > e(3.0,1.0);
  fvar<fvar<var> > f(5.0,1.0);
  fvar<fvar<var> > g(7.0,1.0);

  matrix_ffv a(2,2);
  a << d,e,f,g;

   matrix_d b(2,2);
   b << 2.0, 3.0, 5.0,7.0;
   b = b.inverse();

  matrix_ffv a_inv = stan::agrad::inverse(a);

  EXPECT_NEAR(b(0,0),a_inv(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(b(0,1),a_inv(0,1).val_.val().val(),1.0E-12);
  EXPECT_NEAR(b(1,0),a_inv(1,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(b(1,1),a_inv(1,1).val_.val().val(),1.0E-12);
  EXPECT_NEAR(-8,a_inv(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR( 4,a_inv(0,1).d_.val().val(),1.0E-12);
  EXPECT_NEAR( 6,a_inv(1,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(-3,a_inv(1,1).d_.val().val(),1.0E-12);

  EXPECT_THROW(stan::agrad::inverse(matrix_ffv(2,3)), std::domain_error);

  AVEC q = createAVEC(d.val().val(),e.val().val(),f.val().val(),g.val().val());
  VEC h;
  a_inv(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-49.0,h[0]);
  EXPECT_FLOAT_EQ(35.0,h[1]);
  EXPECT_FLOAT_EQ(21.0,h[2]);
  EXPECT_FLOAT_EQ(-15.0,h[3]);
}
TEST(AgradFwdMatrixInverse,ffv_2ndDeriv_1) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > d(2.0,1.0);
  fvar<fvar<var> > e(3.0,1.0);
  fvar<fvar<var> > f(5.0,1.0);
  fvar<fvar<var> > g(7.0,1.0);

  matrix_ffv a(2,2);
  a << d,e,f,g;

  matrix_ffv a_inv = stan::agrad::inverse(a);

  AVEC q = createAVEC(d.val().val(),e.val().val(),f.val().val(),g.val().val());
  VEC h;
  a_inv(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradFwdMatrixInverse,ffv_2ndDeriv_2) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > d(2.0,1.0);
  fvar<fvar<var> > e(3.0,1.0);
  fvar<fvar<var> > f(5.0,1.0);
  fvar<fvar<var> > g(7.0,1.0);

  matrix_ffv a(2,2);
  a << d,e,f,g;

  matrix_ffv a_inv = stan::agrad::inverse(a);

  AVEC q = createAVEC(d.val().val(),e.val().val(),f.val().val(),g.val().val());
  VEC h;
  a_inv(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-112.0,h[0]);
  EXPECT_FLOAT_EQ(82.0,h[1]);
  EXPECT_FLOAT_EQ(52.0,h[2]);
  EXPECT_FLOAT_EQ(-38.0,h[3]);
}
TEST(AgradFwdMatrixInverse,ffv_3rDeriv) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > d(2.0,1.0);
  fvar<fvar<var> > e(3.0,1.0);
  fvar<fvar<var> > f(5.0,1.0);
  fvar<fvar<var> > g(7.0,1.0);
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;
  f.val_.d_ = 1.0;
  g.val_.d_ = 1.0;

  matrix_ffv a(2,2);
  a << d,e,f,g;

  matrix_ffv a_inv = stan::agrad::inverse(a);

  AVEC q = createAVEC(d.val().val(),e.val().val(),f.val().val(),g.val().val());
  VEC h;
  a_inv(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(-352,h[0]);
  EXPECT_FLOAT_EQ(260,h[1]);
  EXPECT_FLOAT_EQ(168,h[2]);
  EXPECT_FLOAT_EQ(-124,h[3]);
}
