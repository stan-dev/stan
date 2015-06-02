#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/diagonal.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>

TEST(AgradFwdMatrixDiagonal,matrix_fd) {
  using stan::math::diagonal;
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::vector_fd;

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
TEST(AgradFwdMatrixDiagonal,matrix_ffd) {
  using stan::math::diagonal;
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::vector_ffd;
  using stan::math::fvar;

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
