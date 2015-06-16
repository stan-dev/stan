#include <stan/math/fwd/mat/fun/inverse.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>

TEST(AgradFwdMatrixInverse,fd) {
  using stan::math::matrix_fd;
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

  matrix_fd a_inv = stan::math::inverse(a);

  EXPECT_NEAR(b(0,0),a_inv(0,0).val_,1.0E-12);
  EXPECT_NEAR(b(0,1),a_inv(0,1).val_,1.0E-12);
  EXPECT_NEAR(b(1,0),a_inv(1,0).val_,1.0E-12);
  EXPECT_NEAR(b(1,1),a_inv(1,1).val_,1.0E-12);
  EXPECT_NEAR(-8,a_inv(0,0).d_,1.0E-12);
  EXPECT_NEAR( 4,a_inv(0,1).d_,1.0E-12);
  EXPECT_NEAR( 6,a_inv(1,0).d_,1.0E-12);
  EXPECT_NEAR(-3,a_inv(1,1).d_,1.0E-12);

  EXPECT_THROW(stan::math::inverse(matrix_fd(2,3)), std::invalid_argument);
}
TEST(AgradFwdMatrixInverse,ffd) {
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::fvar;

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

  matrix_ffd a_inv = stan::math::inverse(a);

  EXPECT_NEAR(b(0,0),a_inv(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(b(0,1),a_inv(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(b(1,0),a_inv(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(b(1,1),a_inv(1,1).val_.val(),1.0E-12);
  EXPECT_NEAR(-8,a_inv(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR( 4,a_inv(0,1).d_.val(),1.0E-12);
  EXPECT_NEAR( 6,a_inv(1,0).d_.val(),1.0E-12);
  EXPECT_NEAR(-3,a_inv(1,1).d_.val(),1.0E-12);

  EXPECT_THROW(stan::math::inverse(matrix_ffd(2,3)), std::invalid_argument);
}
