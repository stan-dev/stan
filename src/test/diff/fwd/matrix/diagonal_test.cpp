#include <gtest/gtest.h>
#include <stan/math/matrix/diagonal.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

TEST(AgradFwdMatrix,diagonal_matrix) {
  using stan::math::diagonal;
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::vector_fv;

  EXPECT_EQ(0,diagonal(matrix_fv()).size());
  EXPECT_EQ(2,diagonal(matrix_fv(2,2)).size());
  EXPECT_EQ(0,diagonal(matrix_d()).size());
  EXPECT_EQ(2,diagonal(matrix_d(2,2)).size());

  matrix_fv v(3,3);
  v << 1, 4, 9,1, 4, 9,1, 4, 9;
   v(0,0).d_ = 1.0;
   v(1,1).d_ = 2.0;
   v(2,2).d_ = 3.0;
  vector_fv m = diagonal(v);
  EXPECT_EQ(1,m(0).val_);
  EXPECT_EQ(4,m(1).val_);
  EXPECT_EQ(9,m(2).val_);
  EXPECT_EQ(1,m(0).d_);
  EXPECT_EQ(2,m(1).d_);
  EXPECT_EQ(3,m(2).d_);
}
