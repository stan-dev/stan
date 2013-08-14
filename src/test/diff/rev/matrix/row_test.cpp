#include <stan/math/matrix/row.hpp>
#include <gtest/gtest.h>
#include <test/diff/util.hpp>
#include <stan/diff.hpp>
#include <stan/diff/rev/matrix.hpp>

TEST(DiffRevMatrix,row_v) {
  using stan::math::row;
  using stan::diff::matrix_v;
  using stan::diff::vector_v;

  matrix_v y(2,3);
  y << 1, 2, 3, 4, 5, 6;
  vector_v z = row(y,1);
  EXPECT_EQ(3,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val());
  EXPECT_FLOAT_EQ(2.0,z[1].val());
  EXPECT_FLOAT_EQ(3.0,z[2].val());

  vector_v w = row(y,2);
  EXPECT_EQ(3,w.size());
  EXPECT_EQ(4.0,w[0].val());
  EXPECT_EQ(5.0,w[1].val());
  EXPECT_EQ(6.0,w[2].val());
}
TEST(DiffRevMatrix,row_v_exc0) {
  using stan::math::row;
  using stan::diff::matrix_v;

  matrix_v y(2,3);
  y << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(row(y,0),std::domain_error);
  EXPECT_THROW(row(y,7),std::domain_error);
}
TEST(DiffRevMatrix,row_v_excHigh) {
  using stan::math::row;
  using stan::diff::matrix_v;

  matrix_v y(2,3);
  y << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(row(y,0),std::domain_error);
  EXPECT_THROW(row(y,5),std::domain_error);
}
