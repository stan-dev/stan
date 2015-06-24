#include <stan/math/prim/mat/fun/mdivide_left.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,mdivide_left_val) {
  using stan::math::mdivide_left;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_d I;

  Ad << 2.0, 3.0, 
        5.0, 7.0;

  I = mdivide_left(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);
}
