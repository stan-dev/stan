#include <stan/agrad/fwd/matrix/inverse.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/var.hpp>
#include <stan/agrad/rev/matrix/multiply.hpp>

TEST(AgradFwdMatrix,inverse) {
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
TEST(AgradFwdFvarVarMatrix,inverse) {
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
}
TEST(AgradFwdFvarFvarMatrix,inverse) {
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
