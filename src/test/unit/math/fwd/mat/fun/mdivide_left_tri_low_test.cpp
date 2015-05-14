#include <stan/math/fwd/mat/fun/mdivide_left_tri_low.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>

using stan::math::fvar;
TEST(AgradFwdMatrixMdivideLeftTriLow,fd_vector_fd_matrix_fd) {
  using stan::math::matrix_fd;
  using stan::math::vector_fd;

  matrix_fd Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
   Y(2,2).d_ = 2.0;

   vector_fd Z(3);
   Z << 1, 2, 3;
    Z(0).d_ = 2.0;
    Z(1).d_ = 2.0;
    Z(2).d_ = 2.0;

  matrix_fd output = stan::math::mdivide_left_tri_low(Y,Z);

  EXPECT_NEAR(1.0,output(0,0).val_,1.0E-08);
  EXPECT_NEAR(0,output(1,0).val_, 1.0E-08);
  EXPECT_NEAR(-1.0 / 6.0,output(2,0).val_,1.0E-08);
  EXPECT_NEAR(0,output(0,0).d_, 1.0E-08);
  EXPECT_NEAR(0,output(1,0).d_, 1.0E-08);
  EXPECT_NEAR(5.0 / 90.0,output(2,0).d_,1.0E-08);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,fd_vector_matrix_fd) {
  using stan::math::matrix_fd;
  using stan::math::vector_d;

  matrix_fd Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
   Y(2,2).d_ = 2.0;

   vector_d Z(3);
   Z << 1, 2, 3;

  matrix_fd output = stan::math::mdivide_left_tri_low(Y,Z);

  EXPECT_NEAR(1.0,output(0,0).val_, 1.0E-08);
  EXPECT_NEAR(0,output(1,0).val_, 1.0E-08);
  EXPECT_NEAR(-1.0 / 6.0,output(2,0).val_, 1.0E-08);
  EXPECT_NEAR(-2.0,output(0,0).d_, 1.0E-08);
  EXPECT_NEAR(4.0 / 6.0,output(1,0).d_, 1.0E-08);
  EXPECT_NEAR(1.0 / 2.0,output(2,0).d_, 1.0E-08);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,fd_vector_fd_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::vector_fd;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  vector_fd Z(3);
  Z << 1, 2, 3;
   Z(0).d_ = 2.0;
   Z(1).d_ = 2.0;
   Z(2).d_ = 2.0;

  matrix_fd output = stan::math::mdivide_left_tri_low(Y,Z);

  EXPECT_NEAR(1.0,output(0,0).val_, 1.0E-08);
  EXPECT_NEAR(0,output(1,0).val_, 1.0E-08);
  EXPECT_NEAR(-1.0 / 6.0,output(2,0).val_, 1.0E-08);
  EXPECT_NEAR(2.0,output(0,0).d_, 1.0E-08);
  EXPECT_NEAR(-4.0 / 6.0,output(1,0).d_, 1.0E-08);
  EXPECT_NEAR(-4.0 / 9.0,output(2,0).d_, 1.0E-08);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,fd_matrix_fd_matrix_fd) {
  using stan::math::matrix_fd;
  using stan::math::row_vector_fd;

  matrix_fd Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
   Y(2,2).d_ = 2.0;

  matrix_fd Z(3,3);
  Z << 1, 2, 3,
    6, 5, 4,
    7, 8, 9;
   Z(0,0).d_ = 2.0;
   Z(0,1).d_ = 2.0;
   Z(0,2).d_ = 2.0;
   Z(1,0).d_ = 2.0;
   Z(1,1).d_ = 2.0;
   Z(1,2).d_ = 2.0;
   Z(2,0).d_ = 2.0;
   Z(2,1).d_ = 2.0;
   Z(2,2).d_ = 2.0;

  matrix_fd output = stan::math::mdivide_left_tri_low(Z,Y);

  EXPECT_NEAR(1.0,output(0,0).val_, 1.0E-08);
  EXPECT_NEAR(0,output(0,1).val_,1.0E-08);
  EXPECT_NEAR(0,output(0,2).val_, 1.0E-08);
  EXPECT_NEAR(-0.8,output(1,0).val_, 1.0E-08);
  EXPECT_NEAR(0.6,output(1,1).val_, 1.0E-08);
  EXPECT_NEAR(0,output(1,2).val_, 1.0E-08);
  EXPECT_NEAR(34.0 / 90.0,output(2,0).val_, 1.0E-08);
  EXPECT_NEAR(2.0 / 90.0,output(2,1).val_, 1.0E-08);
  EXPECT_NEAR(2.0 / 3.0,output(2,2).val_, 1.0E-08);
  EXPECT_NEAR(0,output(0,0).d_, 1.0E-08);
  EXPECT_NEAR(2.0,output(0,1).d_, 1.0E-08);
  EXPECT_NEAR(2.0,output(0,2).d_, 1.0E-08);
  EXPECT_NEAR(288.0 / 900.0,output(1,0).d_, 1.0E-08);
  EXPECT_NEAR(-2.24,output(1,1).d_, 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,2).d_, 1.0E-08);
  EXPECT_NEAR(-0.19061728,output(2,0).d_, 1.0E-08);
  EXPECT_NEAR(0.5195061728395064,output(2,1).d_, 1.0E-08);
  EXPECT_NEAR(0.2962963,output(2,2).d_, 1.0E-08);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,fd_matrix_fd_matrix) {
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::row_vector_fd;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  matrix_fd Z(3,3);
  Z << 1, 2, 3,
    6, 5, 4,
    7, 8, 9;
   Z(0,0).d_ = 2.0;
   Z(0,1).d_ = 2.0;
   Z(0,2).d_ = 2.0;
   Z(1,0).d_ = 2.0;
   Z(1,1).d_ = 2.0;
   Z(1,2).d_ = 2.0;
   Z(2,0).d_ = 2.0;
   Z(2,1).d_ = 2.0;
   Z(2,2).d_ = 2.0;

  matrix_fd output = stan::math::mdivide_left_tri_low(Z,Y);

  EXPECT_NEAR(1.0,output(0,0).val_, 1.0E-08);
  EXPECT_NEAR(0,output(0,1).val_,1.0E-08);
  EXPECT_NEAR(0,output(0,2).val_,1.0E-08);
  EXPECT_NEAR(-0.8,output(1,0).val_, 1.0E-08);
  EXPECT_NEAR(0.6,output(1,1).val_, 1.0E-08);
  EXPECT_NEAR(0,output(1,2).val_,1.0E-08);
  EXPECT_NEAR(34.0 / 90.0,output(2,0).val_, 1.0E-08);
  EXPECT_NEAR(2.0 / 90.0,output(2,1).val_, 1.0E-08);
  EXPECT_NEAR(2.0 / 3.0,output(2,2).val_, 1.0E-08);
  EXPECT_NEAR(-2.0,output(0,0).d_, 1.0E-08);
  EXPECT_NEAR(0,output(0,1).d_,1.0E-08);
  EXPECT_NEAR(0,output(0,2).d_,1.0E-08);
  EXPECT_NEAR(2.3199999999999985,output(1,0).d_, 1.0E-08);
  EXPECT_NEAR(-0.24,output(1,1).d_, 1.0E-08);
  EXPECT_NEAR(0,output(1,2).d_,1.0E-08);
  EXPECT_NEAR(-0.63506172839506148,output(2,0).d_, 1.0E-08);
  EXPECT_NEAR(0.075061731,output(2,1).d_, 1.0E-08);
  EXPECT_NEAR(-0.14814815,output(2,2).d_, 1.0E-08);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,fd_matrix_matrix_fd) {
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::row_vector_fd;

  matrix_fd Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
   Y(2,2).d_ = 2.0;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    6, 5, 4,
    7, 8, 9;

  matrix_fd output = stan::math::mdivide_left_tri_low(Z,Y);

  EXPECT_NEAR(1.0,output(0,0).val_, 1.0E-08);
  EXPECT_NEAR(0,output(0,1).val_,1.0E-08);
  EXPECT_NEAR(0,output(0,2).val_,1.0E-08);
  EXPECT_NEAR(-0.8,output(1,0).val_, 1.0E-08);
  EXPECT_NEAR(0.6,output(1,1).val_, 1.0E-08);
  EXPECT_NEAR(0,output(1,2).val_,1.0E-08);
  EXPECT_NEAR(34.0 / 90.0,output(2,0).val_, 1.0E-08);
  EXPECT_NEAR(2.0 / 90.0,output(2,1).val_, 1.0E-08);
  EXPECT_NEAR(2.0 / 3.0,output(2,2).val_, 1.0E-08);
  EXPECT_NEAR(2.0,output(0,0).d_, 1.0E-08);
  EXPECT_NEAR(2.0,output(0,1).d_, 1.0E-08);
  EXPECT_NEAR(2.0,output(0,2).d_, 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,0).d_, 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,1).d_, 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,2).d_, 1.0E-08);
  EXPECT_NEAR(4.0 / 9.0,output(2,0).d_, 1.0E-08);
  EXPECT_NEAR(4.0 / 9.0,output(2,1).d_, 1.0E-08);
  EXPECT_NEAR(4.0 / 9.0,output(2,2).d_, 1.0E-08);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,fd_vector_matrix_exceptions) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri_low;

  vector_fd fv1(4), fv2(3);
  vector_d v1(4), v2(3);
  matrix_fd fvm1(4,4), fvm2(3,3);
  matrix_d vm1(4,4), vm2(3,3);

  EXPECT_THROW(mdivide_left_tri_low(fvm2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(vm2,fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm2,v1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(vm1,fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,v2), std::invalid_argument);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,fd_matrix_matrix_exceptions) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::mdivide_left_tri_low;

  matrix_fd fvm1(4,4), fvm2(3,3);
  matrix_d vm1(4,4), vm2(3,3);

  EXPECT_THROW(mdivide_left_tri_low(fvm1,fvm2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,vm2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(vm1,fvm2), std::invalid_argument);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,ffd_vector_ffd_matrix_ffd) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;

  fvar<fvar<double> > a,b,c;

  a.val_.val_ = 1.0;
  b.val_.val_ = 2.0;
  c.val_.val_ = 3.0;
  a.d_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.d_.val_ = 2.0;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
  
  vector_ffd Z(3);
  Z << a,b,c;

  matrix_ffd output = stan::math::mdivide_left_tri_low(Y,Z);

  EXPECT_NEAR(1.0,output(0,0).val_.val(),1.0E-08);
  EXPECT_NEAR(0,output(1,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(-1.0 / 6.0,output(2,0).val_.val(),1.0E-08);
  EXPECT_NEAR(2.0,output(0,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(-2.0 / 3.0, output(1,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(-4.0 / 9.0,output(2,0).d_.val(),1.0E-08);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,ffd_vector_matrix_ffd) {
  using stan::math::matrix_ffd;
  using stan::math::vector_d;

  fvar<fvar<double> > a,b,c,d,e,f,g;

  a.val_.val_ = 1.0;
  b.val_.val_ = 2.0;
  c.val_.val_ = 3.0;
  d.val_.val_ = 4.0;
  e.val_.val_ = 5.0;
  f.val_.val_ = 6.0;
  g.val_.val_ = 0.0;
  a.d_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.d_.val_ = 2.0;
  d.d_.val_ = 2.0;
  e.d_.val_ = 2.0;
  f.d_.val_ = 2.0;
  g.d_.val_ = 2.0;

  matrix_ffd Y(3,3);
  Y << a,g,g,b,c,g,d,e,f;

   vector_d Z(3);
   Z << 1, 2, 3;

  matrix_ffd output = stan::math::mdivide_left_tri_low(Y,Z);

  EXPECT_NEAR(1.0,output(0,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(-1.0 / 6.0,output(2,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(0,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(4.0 / 6.0,output(1,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(1.0 / 2.0,output(2,0).d_.val(), 1.0E-08);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,ffd_vector_ffd_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;

  fvar<fvar<double> > a,b,c;

  a.val_.val_ = 1.0;
  b.val_.val_ = 2.0;
  c.val_.val_ = 3.0;
  a.d_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.d_.val_ = 2.0;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  vector_ffd Z(3);
  Z << a,b,c;

  matrix_ffd output = stan::math::mdivide_left_tri_low(Y,Z);

  EXPECT_NEAR(1.0,output(0,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(-1.0 / 6.0,output(2,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(-4.0 / 6.0,output(1,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(-4.0 / 9.0,output(2,0).d_.val(), 1.0E-08);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,ffd_matrix_ffd_matrix_ffd) {
  using stan::math::matrix_ffd;
  using stan::math::row_vector_ffd;

  fvar<fvar<double> > a,b,c,d,e,f,g,h,i,j;

  a.val_.val_ = 1.0;
  b.val_.val_ = 2.0;
  c.val_.val_ = 3.0;
  d.val_.val_ = 4.0;
  e.val_.val_ = 5.0;
  f.val_.val_ = 6.0;
  h.val_.val_ = 7.0;
  i.val_.val_ = 8.0;
  j.val_.val_ = 9.0;
  g.val_.val_ = 0.0;
  a.d_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.d_.val_ = 2.0;
  d.d_.val_ = 2.0;
  e.d_.val_ = 2.0;
  f.d_.val_ = 2.0;
  g.d_.val_ = 2.0;
  h.d_.val_ = 2.0;
  i.d_.val_ = 2.0;
  j.d_.val_ = 2.0;

  matrix_ffd Y(3,3);
  Y << a,g,g,b,c,g,d,e,f;

  matrix_ffd Z(3,3);
  Z << a,b,c,f,e,d,h,i,j;

  matrix_ffd output = stan::math::mdivide_left_tri_low(Z,Y);

  EXPECT_NEAR(1.0,output(0,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,1).val_.val(),1.0E-08);
  EXPECT_NEAR(0,output(0,2).val_.val(), 1.0E-08);
  EXPECT_NEAR(-0.8,output(1,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0.6,output(1,1).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,2).val_.val(), 1.0E-08);
  EXPECT_NEAR(34.0 / 90.0,output(2,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 90.0,output(2,1).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 3.0,output(2,2).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,2).d_.val(), 1.0E-08);
  EXPECT_NEAR(288.0 / 900.0,output(1,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(-2.24,output(1,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,2).d_.val(), 1.0E-08);
  EXPECT_NEAR(-0.19061728,output(2,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(0.5195061728395064,output(2,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(0.2962963,output(2,2).d_.val(), 1.0E-08);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,ffd_matrix_ffd_matrix) {
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffd;

  fvar<fvar<double> > a,b,c,d,e,f,g,h,i,j;

  a.val_.val_ = 1.0;
  b.val_.val_ = 2.0;
  c.val_.val_ = 3.0;
  d.val_.val_ = 4.0;
  e.val_.val_ = 5.0;
  f.val_.val_ = 6.0;
  h.val_.val_ = 7.0;
  i.val_.val_ = 8.0;
  j.val_.val_ = 9.0;
  g.val_.val_ = 0.0;
  a.d_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.d_.val_ = 2.0;
  d.d_.val_ = 2.0;
  e.d_.val_ = 2.0;
  f.d_.val_ = 2.0;
  g.d_.val_ = 2.0;
  h.d_.val_ = 2.0;
  i.d_.val_ = 2.0;
  j.d_.val_ = 2.0;

  matrix_d Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  matrix_ffd Z(3,3);
  Z << a,b,c,f,e,d,h,i,j;

  matrix_ffd output = stan::math::mdivide_left_tri_low(Z,Y);

  EXPECT_NEAR(1.0,output(0,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,1).val_.val(),1.0E-08);
  EXPECT_NEAR(0,output(0,2).val_.val(),1.0E-08);
  EXPECT_NEAR(-0.8,output(1,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0.6,output(1,1).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,2).val_.val(),1.0E-08);
  EXPECT_NEAR(34.0 / 90.0,output(2,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 90.0,output(2,1).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 3.0,output(2,2).val_.val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(0,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,1).d_.val(),1.0E-08);
  EXPECT_NEAR(0,output(0,2).d_.val(),1.0E-08);
  EXPECT_NEAR(2.3199999999999985,output(1,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(-0.24,output(1,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,2).d_.val(),1.0E-08);
  EXPECT_NEAR(-0.63506172839506148,output(2,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(0.075061731,output(2,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(-0.14814815,output(2,2).d_.val(), 1.0E-08);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,ffd_matrix_matrix_ffd) {
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffd;

  fvar<fvar<double> > a,b,c,d,e,f,g;

  a.val_.val_ = 1.0;
  b.val_.val_ = 2.0;
  c.val_.val_ = 3.0;
  d.val_.val_ = 4.0;
  e.val_.val_ = 5.0;
  f.val_.val_ = 6.0;
  g.val_.val_ = 0.0;
  a.d_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.d_.val_ = 2.0;
  d.d_.val_ = 2.0;
  e.d_.val_ = 2.0;
  f.d_.val_ = 2.0;
  g.d_.val_ = 2.0;

  matrix_ffd Y(3,3);
  Y << a,g,g,b,c,g,d,e,f;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    6, 5, 4,
    7, 8, 9;

  matrix_ffd output = stan::math::mdivide_left_tri_low(Z,Y);

  EXPECT_NEAR(1.0,output(0,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(0,1).val_.val(),1.0E-08);
  EXPECT_NEAR(0,output(0,2).val_.val(),1.0E-08);
  EXPECT_NEAR(-0.8,output(1,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(0.6,output(1,1).val_.val(), 1.0E-08);
  EXPECT_NEAR(0,output(1,2).val_.val(),1.0E-08);
  EXPECT_NEAR(34.0 / 90.0,output(2,0).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 90.0,output(2,1).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0 / 3.0,output(2,2).val_.val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(2.0,output(0,2).d_.val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(-2.0,output(1,2).d_.val(), 1.0E-08);
  EXPECT_NEAR(4.0 / 9.0,output(2,0).d_.val(), 1.0E-08);
  EXPECT_NEAR(4.0 / 9.0,output(2,1).d_.val(), 1.0E-08);
  EXPECT_NEAR(4.0 / 9.0,output(2,2).d_.val(), 1.0E-08);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,ffd_vector_matrix_exceptions) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  using stan::math::vector_d;
  using stan::math::mdivide_left_tri_low;

  vector_ffd fv1(4), fv2(3);
  vector_d v1(4), v2(3);
  matrix_ffd fvm1(4,4), fvm2(3,3);
  matrix_d vm1(4,4), vm2(3,3);

  EXPECT_THROW(mdivide_left_tri_low(fvm2, fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(vm2,fv1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm2,v1), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(vm1,fv2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,v2), std::invalid_argument);
}
TEST(AgradFwdMatrixMdivideLeftTriLow,ffd_matrix_matrix_exceptions) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::mdivide_left_tri_low;

  matrix_ffd fvm1(4,4), fvm2(3,3);
  matrix_d vm1(4,4), vm2(3,3);

  EXPECT_THROW(mdivide_left_tri_low(fvm1,fvm2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(fvm1,vm2), std::invalid_argument);
  EXPECT_THROW(mdivide_left_tri_low(vm1,fvm2), std::invalid_argument);
}
