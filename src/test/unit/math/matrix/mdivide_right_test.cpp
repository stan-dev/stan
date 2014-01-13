#include <stan/math/matrix/mdivide_right.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,mdivide_right_val) {
  using stan::math::mdivide_right;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_d I;

  Ad << 2.0, 3.0, 
        5.0, 7.0;

  I = mdivide_right(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);
}

TEST(MathMatrix,mdivide_right_val2) {
  using stan::math::mdivide_right;
  stan::math::row_vector_d b(5);
  stan::math::matrix_d A(5,5);
  stan::math::row_vector_d expected(5);
  stan::math::row_vector_d x;

  b << 19, 150, -170, 140, 31;
  A << 
    1, 8, -9, 7, 5, 
    0, 1, 0, 4, 4, 
    0, 0, 1, 2, 5, 
    0, 0, 0, 1, -5, 
    0, 0, 0, 0, 1;
  expected << 19, -2, 1, 13, 4;
  x = mdivide_right(b, A);
  
  ASSERT_EQ(expected.size(), x.size());
  for (int n = 0; n < expected.size(); n++)
    EXPECT_FLOAT_EQ(expected(n), x(n));
}
