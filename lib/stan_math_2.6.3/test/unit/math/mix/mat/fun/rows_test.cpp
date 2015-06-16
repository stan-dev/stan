#include <stan/math/prim/mat/fun/rows.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>

TEST(AgradMixMatrixRows,ffd_vector) {
  using stan::math::vector_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::rows;

  vector_ffd v(5);
  v << 0, 1, 2, 3, 4;
  EXPECT_EQ(5U, rows(v));
  
  v.resize(0);
  EXPECT_EQ(0U, rows(v));
}
TEST(AgradMixMatrixRows,ffd_rowvector) {
  using stan::math::row_vector_ffd;
  using stan::math::rows;

  row_vector_ffd rv(5);
  rv << 0, 1, 2, 3, 4;
  EXPECT_EQ(1U, rows(rv));

  rv.resize(0);
  EXPECT_EQ(1U, rows(rv));
}
TEST(AgradMixMatrixRows,ffd_matrix) {
  using stan::math::matrix_ffd;
  using stan::math::rows;

  matrix_ffd m(2,3);
  m << 0, 1, 2, 3, 4, 5;
  EXPECT_EQ(2U, rows(m));
  
  m.resize(0,2);
  EXPECT_EQ(0U, rows(m));
}
