#include <stan/agrad/fwd/matrix/determinant.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>

TEST(AgradFwdMatrix,determinant) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;

  matrix_fv a(2,2);
  a << 2.0, 3.0, 5.0, 7.0;
   a(0,0).d_ = 1.0;
   a(0,1).d_ = 1.0;
   a(1,0).d_ = 1.0;
   a(1,1).d_ = 1.0;

   fvar<double> a_det = stan::agrad::determinant(a);

   EXPECT_FLOAT_EQ(-1,a_det.val_);
   EXPECT_FLOAT_EQ(1,a_det.d_);

  EXPECT_THROW(determinant(matrix_fv(2,3)), std::domain_error);
}
