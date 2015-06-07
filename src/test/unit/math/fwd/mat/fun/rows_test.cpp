#include <stan/math/prim/mat/fun/rows.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>

TEST(AgradFwdMatrixRows,fd_vector) {
  using stan::math::vector_fd;
  using stan::math::row_vector_fd;
  using stan::math::rows;

  vector_fd v(5);
  v << 0, 1, 2, 3, 4;
  EXPECT_EQ(5U, rows(v));
  
  v.resize(0);
  EXPECT_EQ(0U, rows(v));
}
TEST(AgradFwdMatrixRows,fd_rowvector) {
  using stan::math::row_vector_fd;
  using stan::math::rows;

  row_vector_fd rv(5);
  rv << 0, 1, 2, 3, 4;
  EXPECT_EQ(1U, rows(rv));

  rv.resize(0);
  EXPECT_EQ(1U, rows(rv));
}
TEST(AgradFwdMatrixRows,fd_matrix) {
  using stan::math::matrix_fd;
  using stan::math::rows;

  matrix_fd m(2,3);
  m << 0, 1, 2, 3, 4, 5;
  EXPECT_EQ(2U, rows(m));
  
  m.resize(0,2);
  EXPECT_EQ(0U, rows(m));
}
TEST(AgradFwdMatrixRows,ffd_vector) {
  using stan::math::vector_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::rows;

  vector_ffd v(5);
  v << 0, 1, 2, 3, 4;
  EXPECT_EQ(5U, rows(v));
  
  v.resize(0);
  EXPECT_EQ(0U, rows(v));
}
TEST(AgradFwdMatrixRows,ffd_rowvector) {
  using stan::math::row_vector_ffd;
  using stan::math::rows;

  row_vector_ffd rv(5);
  rv << 0, 1, 2, 3, 4;
  EXPECT_EQ(1U, rows(rv));

  rv.resize(0);
  EXPECT_EQ(1U, rows(rv));
}
TEST(AgradFwdMatrixRows,ffd_matrix) {
  using stan::math::matrix_ffd;
  using stan::math::rows;

  matrix_ffd m(2,3);
  m << 0, 1, 2, 3, 4, 5;
  EXPECT_EQ(2U, rows(m));
  
  m.resize(0,2);
  EXPECT_EQ(0U, rows(m));
}
