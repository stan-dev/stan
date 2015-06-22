#include <stan/math/prim/mat/fun/row.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,row) {
  stan::math::matrix_d m(3,4);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  stan::math::row_vector_d c = m.row(1);
  stan::math::row_vector_d c2 = stan::math::row(m,2);
  EXPECT_EQ(4,c.size());
  EXPECT_EQ(4,c2.size());
  for (size_t i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(c[i],c2[i]);
}

TEST(MathMatrix, row_exception) {
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  
  using stan::math::row;

  EXPECT_THROW(row(m1,5),std::out_of_range);
  EXPECT_THROW(row(m1,0),std::out_of_range);
}
