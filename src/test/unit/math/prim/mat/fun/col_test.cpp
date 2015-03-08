#include <stan/math/prim/mat/fun/col.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, col) {
  stan::math::matrix_d m(3,4);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  stan::math::vector_d c = m.col(0);
  stan::math::vector_d c2 = stan::math::col(m,1);
  EXPECT_EQ(3,c.size());
  EXPECT_EQ(3,c2.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(c[i],c2[i]);
}

TEST(MathMatrix, col_exception) {
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::col;
  EXPECT_THROW(col(m1,5),std::out_of_range);
  EXPECT_THROW(col(m1,0),std::out_of_range);

}
