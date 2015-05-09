#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/diag_matrix.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>

TEST(AgradFwdMatrixDiagMatrix,vector_fd) {
  using stan::math::diag_matrix;
  using stan::math::matrix_fd;
  using stan::math::vector_d;
  using stan::math::vector_fd;

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
TEST(AgradFwdMatrixDiagMatrix,vector_ffd) {
  using stan::math::diag_matrix;
  using stan::math::matrix_ffd;
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::fvar;

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
