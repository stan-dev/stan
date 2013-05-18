#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/math/matrix/diag_matrix.hpp>

TEST(AgradFwdMatrix,diagMatrix) {
  using stan::math::diag_matrix;
  using stan::agrad::matrix_fv;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  EXPECT_EQ(0,diag_matrix(vector_fv()).size());
  EXPECT_EQ(4,diag_matrix(vector_fv(2)).size());
  EXPECT_EQ(0,diag_matrix(vector_d()).size());
  EXPECT_EQ(4,diag_matrix(vector_d(2)).size());

  vector_fv v(3);
  v << 1, 4, 9;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
  matrix_fv m = diag_matrix(v);
  EXPECT_EQ(1,m(0,0).val_);
  EXPECT_EQ(4,m(1,1).val_);
  EXPECT_EQ(9,m(2,2).val_);
  EXPECT_EQ(1,m(0,0).d_);
  EXPECT_EQ(1,m(1,1).d_);
  EXPECT_EQ(1,m(2,2).d_);
}
