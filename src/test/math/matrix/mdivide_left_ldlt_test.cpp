#include <stan/math/matrix/mdivide_left_ldlt.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,mdivide_left_ldlt_val) {
  stan::math::LDLT_factor<double,-1,-1> ldlt_Ad;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_d I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;

  ldlt_Ad.compute(Ad);

  I = mdivide_left_ldlt(ldlt_Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);
}

