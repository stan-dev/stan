#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/math/matrix/diag_matrix.hpp>
#include <stan/agrad/var.hpp>

TEST(AgradFwdMatrix,diagMatrix) {
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
TEST(AgradFwdFvarVarMatrix,diagMatrix) {
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
}
TEST(AgradFwdFvarFvarMatrix,diagMatrix) {
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
