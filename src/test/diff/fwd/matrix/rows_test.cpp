#include <stan/math/matrix/rows.hpp>
#include <gtest/gtest.h>
#include <stan/diff/fwd/matrix/typedefs.hpp>

TEST(DiffFwdMatrix,rows_vector) {
  using stan::diff::vector_fv;
  using stan::diff::row_vector_fv;
  using stan::math::rows;

  vector_fv v(5);
  v << 0, 1, 2, 3, 4;
  EXPECT_EQ(5U, rows(v));
  
  v.resize(0);
  EXPECT_EQ(0U, rows(v));
}
TEST(DiffFwdMatrix,rows_rowvector) {
  using stan::diff::row_vector_fv;
  using stan::math::rows;

  row_vector_fv rv(5);
  rv << 0, 1, 2, 3, 4;
  EXPECT_EQ(1U, rows(rv));

  rv.resize(0);
  EXPECT_EQ(1U, rows(rv));
}
TEST(DiffFwdMatrix,rows_matrix) {
  using stan::diff::matrix_fv;
  using stan::math::rows;

  matrix_fv m(2,3);
  m << 0, 1, 2, 3, 4, 5;
  EXPECT_EQ(2U, rows(m));
  
  m.resize(0,2);
  EXPECT_EQ(0U, rows(m));
}
