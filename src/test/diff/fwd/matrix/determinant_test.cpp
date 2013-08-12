#include <stan/diff/fwd/matrix/determinant.hpp>
#include <gtest/gtest.h>
#include <stan/diff/fwd/matrix/typedefs.hpp>
#include <stan/diff/fwd/fvar.hpp>

TEST(AgradFwdMatrix,determinant) {
  using stan::diff::matrix_fv;
  using stan::math::matrix_d;
  using stan::diff::fvar;

  matrix_fv a(2,2);
  a << 2.0, 3.0, 5.0, 7.0;
   a(0,0).d_ = 1.0;
   a(0,1).d_ = 1.0;
   a(1,0).d_ = 1.0;
   a(1,1).d_ = 1.0;

   fvar<double> a_det = stan::diff::determinant(a);

   EXPECT_FLOAT_EQ(-1,a_det.val_);
   EXPECT_FLOAT_EQ(1,a_det.d_);

  EXPECT_THROW(determinant(matrix_fv(2,3)), std::domain_error);
}
