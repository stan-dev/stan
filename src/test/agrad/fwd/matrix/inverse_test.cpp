#include <stan/agrad/fwd/matrix/inverse.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

TEST(AgradFwdMatrix,inverse) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;

  matrix_fv a(2,2);
  a << 2.0, 3.0, 5.0, 7.0;
   a(0,0).d_ = 1.0;
   a(0,1).d_ = 1.0;
   a(1,0).d_ = 1.0;
   a(1,1).d_ = 1.0;

   matrix_d b(2,2);
   b << 2.0, 3.0, 5.0,7.0;
   b = b.inverse();

  matrix_fv a_inv = stan::agrad::inverse(a);

  EXPECT_NEAR(b(0,0),a_inv(0,0).val_,1.0E-12);
  EXPECT_NEAR(b(0,1),a_inv(0,1).val_,1.0E-12);
  EXPECT_NEAR(b(1,0),a_inv(1,0).val_,1.0E-12);
  EXPECT_NEAR(b(1,1),a_inv(1,1).val_,1.0E-12);
  EXPECT_NEAR(-8,a_inv(0,0).d_,1.0E-12);
  EXPECT_NEAR( 4,a_inv(0,1).d_,1.0E-12);
  EXPECT_NEAR( 6,a_inv(1,0).d_,1.0E-12);
  EXPECT_NEAR(-3,a_inv(1,1).d_,1.0E-12);

  EXPECT_THROW(stan::agrad::inverse(matrix_fv(2,3)), std::domain_error);
}
