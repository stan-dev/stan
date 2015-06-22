#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/diag_matrix.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>

TEST(AgradMixMatrixDiagMatrix,vector_fv_1stDeriv) {
  using stan::math::diag_matrix;
  using stan::math::matrix_fv;
  using stan::math::vector_d;
  using stan::math::vector_fv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixDiagMatrix,vector_fv_2ndDeriv) {
  using stan::math::diag_matrix;
  using stan::math::matrix_fv;
  using stan::math::vector_d;
  using stan::math::vector_fv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixDiagMatrix,vector_ffv_1stDeriv) {
  using stan::math::diag_matrix;
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixDiagMatrix,vector_ffv_2ndDeriv_1) {
  using stan::math::diag_matrix;
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixDiagMatrix,vector_ffv_2ndDeriv_2) {
  using stan::math::diag_matrix;
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixMatrixDiagMatrix,vector_ffv_3rdDeriv) {
  using stan::math::diag_matrix;
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::fvar;
  using stan::math::var;

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

