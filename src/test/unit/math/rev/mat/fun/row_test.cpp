#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/row.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>

TEST(AgradRevMatrix,row_v) {
  using stan::math::row;
  using stan::math::matrix_v;
  using stan::math::vector_v;

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
TEST(AgradRevMatrix,row_v_exc0) {
  using stan::math::row;
  using stan::math::matrix_v;

  matrix_v y(2,3);
  y << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(row(y,0),std::out_of_range);
  EXPECT_THROW(row(y,7),std::out_of_range);
}
TEST(AgradRevMatrix,row_v_excHigh) {
  using stan::math::row;
  using stan::math::matrix_v;

  matrix_v y(2,3);
  y << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(row(y,0),std::out_of_range);
  EXPECT_THROW(row(y,5),std::out_of_range);
}
