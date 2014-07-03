#include <gtest/gtest.h>
#include <stan/math/matrix/diag_matrix.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixDiagMatrix,vector_fd) {
  using stan::math::diag_matrix;
  using stan::agrad::matrix_fd;
  using stan::math::vector_d;
  using stan::agrad::vector_fd;

  EXPECT_EQ(0,diag_matrix(vector_fd()).size());
  EXPECT_EQ(4,diag_matrix(vector_fd(2)).size());
  EXPECT_EQ(0,diag_matrix(vector_d()).size());
  EXPECT_EQ(4,diag_matrix(vector_d(2)).size());

  vector_fd v(3);
  v << 1, 4, 9;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
  matrix_fd m = diag_matrix(v);
  EXPECT_EQ(1,m(0,0).val_);
  EXPECT_EQ(4,m(1,1).val_);
  EXPECT_EQ(9,m(2,2).val_);
  EXPECT_EQ(1,m(0,0).d_);
  EXPECT_EQ(1,m(1,1).d_);
  EXPECT_EQ(1,m(2,2).d_);
}
TEST(AgradFwdMatrixDiagMatrix,vector_fv_1stDeriv) {
  using stan::math::diag_matrix;
  using stan::agrad::matrix_fv;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  EXPECT_EQ(0,diag_matrix(vector_fv()).size());
  EXPECT_EQ(4,diag_matrix(vector_fv(2)).size());
  EXPECT_EQ(0,diag_matrix(vector_d()).size());
  EXPECT_EQ(4,diag_matrix(vector_d(2)).size());

  fvar<var> a(1.0,1.0);
  fvar<var> b(4.0,1.0);
  fvar<var> c(9.0,1.0);

  vector_fv v(3);
  v << a,b,c;
  matrix_fv m = diag_matrix(v);
  EXPECT_EQ(1,m(0,0).val_.val());
  EXPECT_EQ(4,m(1,1).val_.val());
  EXPECT_EQ(9,m(2,2).val_.val());
  EXPECT_EQ(1,m(0,0).d_.val());
  EXPECT_EQ(1,m(1,1).d_.val());
  EXPECT_EQ(1,m(2,2).d_.val());

  AVEC z = createAVEC(a.val(),b.val(),c.val());
  VEC h;
  m(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradFwdMatrixDiagMatrix,vector_fv_2ndDeriv) {
  using stan::math::diag_matrix;
  using stan::agrad::matrix_fv;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(4.0,1.0);
  fvar<var> c(9.0,1.0);

  vector_fv v(3);
  v << a,b,c;
  matrix_fv m = diag_matrix(v);

  AVEC z = createAVEC(a.val(),b.val(),c.val());
  VEC h;
  m(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradFwdMatrixDiagMatrix,vector_ffd) {
  using stan::math::diag_matrix;
  using stan::agrad::matrix_ffd;
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;
  using stan::agrad::fvar;

  EXPECT_EQ(0,diag_matrix(vector_ffd()).size());
  EXPECT_EQ(4,diag_matrix(vector_ffd(2)).size());
  EXPECT_EQ(0,diag_matrix(vector_d()).size());
  EXPECT_EQ(4,diag_matrix(vector_d(2)).size());

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 4.0;
  b.d_.val_ = 1.0;
  c.val_.val_ = 9.0;
  c.d_.val_ = 1.0;

  vector_ffd v(3);
  v << a,b,c;
  matrix_ffd m = diag_matrix(v);
  EXPECT_EQ(1,m(0,0).val_.val());
  EXPECT_EQ(4,m(1,1).val_.val());
  EXPECT_EQ(9,m(2,2).val_.val());
  EXPECT_EQ(1,m(0,0).d_.val());
  EXPECT_EQ(1,m(1,1).d_.val());
  EXPECT_EQ(1,m(2,2).d_.val());
}
TEST(AgradFwdMatrixDiagMatrix,vector_ffv_1stDeriv) {
  using stan::math::diag_matrix;
  using stan::agrad::matrix_ffv;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  EXPECT_EQ(0,diag_matrix(vector_ffv()).size());
  EXPECT_EQ(4,diag_matrix(vector_ffv(2)).size());
  EXPECT_EQ(0,diag_matrix(vector_d()).size());
  EXPECT_EQ(4,diag_matrix(vector_d(2)).size());

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(4.0,1.0);
  fvar<fvar<var> > c(9.0,1.0);

  vector_ffv v(3);
  v << a,b,c;
  matrix_ffv m = diag_matrix(v);
  EXPECT_EQ(1,m(0,0).val_.val().val());
  EXPECT_EQ(4,m(1,1).val_.val().val());
  EXPECT_EQ(9,m(2,2).val_.val().val());
  EXPECT_EQ(1,m(0,0).d_.val().val());
  EXPECT_EQ(1,m(1,1).d_.val().val());
  EXPECT_EQ(1,m(2,2).d_.val().val());

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  m(0,0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradFwdMatrixDiagMatrix,vector_ffv_2ndDeriv_1) {
  using stan::math::diag_matrix;
  using stan::agrad::matrix_ffv;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(4.0,1.0);
  fvar<fvar<var> > c(9.0,1.0);

  vector_ffv v(3);
  v << a,b,c;
  matrix_ffv m = diag_matrix(v);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  m(0,0).val().d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradFwdMatrixDiagMatrix,vector_ffv_2ndDeriv_2) {
  using stan::math::diag_matrix;
  using stan::agrad::matrix_ffv;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(4.0,1.0);
  fvar<fvar<var> > c(9.0,1.0);

  vector_ffv v(3);
  v << a,b,c;
  matrix_ffv m = diag_matrix(v);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  m(0,0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradFwdMatrixDiagMatrix,vector_ffv_3rdDeriv) {
  using stan::math::diag_matrix;
  using stan::agrad::matrix_ffv;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(4.0,1.0);
  fvar<fvar<var> > c(9.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;

  vector_ffv v(3);
  v << a,b,c;
  matrix_ffv m = diag_matrix(v);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  m(0,0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}

