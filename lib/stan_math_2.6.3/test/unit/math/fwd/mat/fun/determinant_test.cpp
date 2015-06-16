#include <stan/math/fwd/mat/fun/determinant.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>

TEST(AgradFwdMatrixDeterminant,matrix_fd) {
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::fvar;
  
  matrix_fd a(2,2);
  a << 2.0, 3.0, 5.0, 7.0;
   a(0,0).d_ = 1.0;
   a(0,1).d_ = 1.0;
   a(1,0).d_ = 1.0;
   a(1,1).d_ = 1.0;

  fvar<double> a_det = stan::math::determinant(a);
   
  EXPECT_FLOAT_EQ(-1,a_det.val_);
  EXPECT_FLOAT_EQ(1,a_det.d_);

  EXPECT_THROW(determinant(matrix_fd(2,3)), std::invalid_argument);
}
TEST(AgradFwdMatrixDeterminant,matrix_ffd) {
  using stan::math::matrix_ffd;
  using stan::math::fvar;

  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 2.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 3.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 5.0;
  c.d_.val_ = 1.0;
  d.val_.val_ = 7.0;
  d.d_.val_ = 1.0; 

  matrix_ffd g(2,2);
  g << a,b,c,d;

  fvar<fvar<double> > a_det = stan::math::determinant(g);

   EXPECT_FLOAT_EQ(-1,a_det.val_.val());
   EXPECT_FLOAT_EQ(1,a_det.d_.val());

  EXPECT_THROW(determinant(matrix_ffd(2,3)), std::invalid_argument);
}
