#include <gtest/gtest.h>
#include <stan/math/matrix/diagonal.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/var.hpp>

TEST(AgradFwdMatrix,diagonal_matrix) {
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
TEST(AgradFwdFvarVarMatrix,diagonal_matrix) {
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
}
TEST(AgradFwdFvarFvarMatrix,diagonal_matrix) {
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
