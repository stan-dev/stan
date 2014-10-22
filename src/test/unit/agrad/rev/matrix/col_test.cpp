#include <stan/math/matrix/col.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>

TEST(AgradRevMatrix,col_v) {
  using stan::math::col;
  using stan::agrad::matrix_v;
  using stan::agrad::vector_v;

  matrix_v y(2,3);
  y << 1, 2, 3, 4, 5, 6;
  vector_v z = col(y,1);
  EXPECT_EQ(2,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val());
  EXPECT_FLOAT_EQ(4.0,z[1].val());

  vector_v w = col(y,2);
  EXPECT_EQ(2,w.size());
  EXPECT_EQ(2.0,w[0].val());
  EXPECT_EQ(5.0,w[1].val());
}
TEST(AgradRevMatrix,col_v_exc0) {
  using stan::math::col;
  using stan::agrad::matrix_v;

  matrix_v y(2,3);
  y << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(col(y,0),std::domain_error);
  EXPECT_THROW(col(y,7),std::domain_error);
}
TEST(AgradRevMatrix,col_v_excHigh) {
  using stan::math::col;
  using stan::agrad::matrix_v;

  matrix_v y(2,3);
  y << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(col(y,0),std::domain_error);
  EXPECT_THROW(col(y,5),std::domain_error);
}
