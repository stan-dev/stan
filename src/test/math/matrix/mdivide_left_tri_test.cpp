#include <stan/math/matrix/mdivide_left_tri.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,mdivide_left_tri_val) {
  using stan::math::mdivide_left_tri;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_d Ad_inv(2,2);
  stan::math::matrix_d I;

  Ad << 2.0, 0.0, 
        5.0, 7.0;

  I = mdivide_left_tri<Eigen::Lower>(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);

  Ad_inv = mdivide_left_tri<Eigen::Lower>(Ad);
  I = Ad * Ad_inv;
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);

  Ad << 2.0, 3.0, 
        0.0, 7.0;

  I = mdivide_left_tri<Eigen::Upper>(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);
}
