#include <stan/math/matrix/rows.hpp>
#include <gtest/gtest.h>
#include <test/diff/util.hpp>
#include <stan/diff/rev/matrix/typedefs.hpp>

TEST(DiffRevMatrix,rows_vector) {
  using stan::diff::vector_v;
  using stan::diff::row_vector_v;
  using stan::math::rows;

  vector_v v(5);
  v << 0, 1, 2, 3, 4;
  EXPECT_EQ(5U, rows(v));
  
  v.resize(0);
  EXPECT_EQ(0U, rows(v));
}
TEST(DiffRevMatrix,rows_rowvector) {
  using stan::diff::row_vector_v;
  using stan::math::rows;

  row_vector_v rv(5);
  rv << 0, 1, 2, 3, 4;
  EXPECT_EQ(1U, rows(rv));

  rv.resize(0);
  EXPECT_EQ(1U, rows(rv));
}
TEST(DiffRevMatrix,rows_matrix) {
  using stan::diff::matrix_v;
  using stan::math::rows;

  matrix_v m(2,3);
  m << 0, 1, 2, 3, 4, 5;
  EXPECT_EQ(2U, rows(m));
  
  m.resize(0,2);
  EXPECT_EQ(0U, rows(m));
}
