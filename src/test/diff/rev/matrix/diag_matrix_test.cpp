#include <gtest/gtest.h>
#include <test/diff/util.hpp>
#include <stan/diff.hpp>
#include <stan/diff/rev/matrix.hpp>

TEST(DiffRevMatrix,diagMatrix) {
  using stan::math::diag_matrix;
  using stan::diff::matrix_v;
  using stan::math::vector_d;
  using stan::diff::vector_v;

  EXPECT_EQ(0,diag_matrix(vector_v()).size());
  EXPECT_EQ(4,diag_matrix(vector_v(2)).size());
  EXPECT_EQ(0,diag_matrix(vector_d()).size());
  EXPECT_EQ(4,diag_matrix(vector_d(2)).size());

  vector_v v(3);
  v << 1, 4, 9;
  matrix_v m = diag_matrix(v);
  EXPECT_EQ(1,m(0,0).val());
  EXPECT_EQ(4,m(1,1).val());
  EXPECT_EQ(9,m(2,2).val());
}
