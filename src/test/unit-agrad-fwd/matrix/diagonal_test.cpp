#include <gtest/gtest.h>
#include <stan/math/matrix/diagonal.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixDiagonal,matrix_fd) {
  using stan::math::diagonal;
  using stan::agrad::matrix_fd;
  using stan::math::matrix_d;
  using stan::agrad::vector_fd;

  EXPECT_EQ(0,diagonal(matrix_fd()).size());
  EXPECT_EQ(2,diagonal(matrix_fd(2,2)).size());
  EXPECT_EQ(0,diagonal(matrix_d()).size());
  EXPECT_EQ(2,diagonal(matrix_d(2,2)).size());

  matrix_fd v(3,3);
  v << 1, 4, 9,1, 4, 9,1, 4, 9;
   v(0,0).d_ = 1.0;
   v(1,1).d_ = 2.0;
   v(2,2).d_ = 3.0;
  vector_fd m = diagonal(v);
  EXPECT_EQ(1,m(0).val_);
  EXPECT_EQ(4,m(1).val_);
  EXPECT_EQ(9,m(2).val_);
  EXPECT_EQ(1,m(0).d_);
  EXPECT_EQ(2,m(1).d_);
  EXPECT_EQ(3,m(2).d_);
}
TEST(AgradFwdMatrixDiagonal,matrix_fv_1stDeriv) {
  using stan::math::diagonal;
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  EXPECT_EQ(0,diagonal(matrix_fv()).size());
  EXPECT_EQ(2,diagonal(matrix_fv(2,2)).size());
  EXPECT_EQ(0,diagonal(matrix_d()).size());
  EXPECT_EQ(2,diagonal(matrix_d(2,2)).size());

  fvar<var> a(1.0,1.0);
  fvar<var> b(4.0,2.0);
  fvar<var> c(9.0,3.0);
  matrix_fv v(3,3);
  v << a,b,c,a,b,c,a,b,c;

  vector_fv m = diagonal(v);
  EXPECT_EQ(1,m(0).val_.val());
  EXPECT_EQ(4,m(1).val_.val());
  EXPECT_EQ(9,m(2).val_.val());
  EXPECT_EQ(1,m(0).d_.val());
  EXPECT_EQ(2,m(1).d_.val());
  EXPECT_EQ(3,m(2).d_.val());

  AVEC z = createAVEC(a.val(),b.val(),c.val());
  VEC h;
  m(0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradFwdMatrixDiagonal,matrix_fv_2ndDeriv) {
  using stan::math::diagonal;
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(4.0,2.0);
  fvar<var> c(9.0,3.0);
  matrix_fv v(3,3);
  v << a,b,c,a,b,c,a,b,c;

  vector_fv m = diagonal(v);

  AVEC z = createAVEC(a.val(),b.val(),c.val());
  VEC h;
  m(0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradFwdMatrixDiagonal,matrix_ffd) {
  using stan::math::diagonal;
  using stan::agrad::matrix_ffd;
  using stan::math::matrix_d;
  using stan::agrad::vector_ffd;
  using stan::agrad::fvar;

  EXPECT_EQ(0,diagonal(matrix_ffd()).size());
  EXPECT_EQ(2,diagonal(matrix_ffd(2,2)).size());
  EXPECT_EQ(0,diagonal(matrix_d()).size());
  EXPECT_EQ(2,diagonal(matrix_d(2,2)).size());

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 4.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 9.0;
  c.d_.val_ = 3.0;
  matrix_ffd v(3,3);
  v << a,b,c,a,b,c,a,b,c;

  vector_ffd m = diagonal(v);
  EXPECT_EQ(1,m(0).val_.val());
  EXPECT_EQ(4,m(1).val_.val());
  EXPECT_EQ(9,m(2).val_.val());
  EXPECT_EQ(1,m(0).d_.val());
  EXPECT_EQ(2,m(1).d_.val());
  EXPECT_EQ(3,m(2).d_.val());
}

TEST(AgradFwdMatrixDiagonal,matrix_ffv_1stDeriv) {
  using stan::math::diagonal;
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  EXPECT_EQ(0,diagonal(matrix_ffv()).size());
  EXPECT_EQ(2,diagonal(matrix_ffv(2,2)).size());
  EXPECT_EQ(0,diagonal(matrix_d()).size());
  EXPECT_EQ(2,diagonal(matrix_d(2,2)).size());

  fvar<fvar<var> >  a(1.0,1.0);
  fvar<fvar<var> >  b(4.0,2.0);
  fvar<fvar<var> >  c(9.0,3.0);
  matrix_ffv v(3,3);
  v << a,b,c,a,b,c,a,b,c;

  vector_ffv m = diagonal(v);
  EXPECT_EQ(1,m(0).val_.val().val());
  EXPECT_EQ(4,m(1).val_.val().val());
  EXPECT_EQ(9,m(2).val_.val().val());
  EXPECT_EQ(1,m(0).d_.val().val());
  EXPECT_EQ(2,m(1).d_.val().val());
  EXPECT_EQ(3,m(2).d_.val().val());

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  m(0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradFwdMatrixDiagonal,matrix_ffv_2ndDeriv_1) {
  using stan::math::diagonal;
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(1.0,1.0);
  fvar<fvar<var> >  b(4.0,2.0);
  fvar<fvar<var> >  c(9.0,3.0);
  matrix_ffv v(3,3);
  v << a,b,c,a,b,c,a,b,c;

  vector_ffv m = diagonal(v);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  m(0).val().d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradFwdMatrixDiagonal,matrix_ffv_2ndDeriv_2) {
  using stan::math::diagonal;
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(1.0,1.0);
  fvar<fvar<var> >  b(4.0,2.0);
  fvar<fvar<var> >  c(9.0,3.0);
  matrix_ffv v(3,3);
  v << a,b,c,a,b,c,a,b,c;

  vector_ffv m = diagonal(v);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  m(0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}

TEST(AgradFwdMatrixDiagonal,matrix_ffv_3rdDeriv) {
  using stan::math::diagonal;
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> >  a(1.0,1.0);
  fvar<fvar<var> >  b(4.0,1.0);
  fvar<fvar<var> >  c(9.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;

  matrix_ffv v(3,3);
  v << a,b,c,a,b,c,a,b,c;

  vector_ffv m = diagonal(v);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  m(0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
