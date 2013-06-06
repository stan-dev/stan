#include <stan/agrad/fwd/matrix/determinant.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

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
TEST(AgradFwdFvarVarMatrix,determinant) {
  using stan::agrad::matrix_fvv;
  using stan::math::matrix_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(5.0,1.0);
  fvar<var> e(7.0,1.0);

  matrix_fvv a(2,2);
  a << b,c,d,e;

   fvar<var> a_det = stan::agrad::determinant(a);

   EXPECT_FLOAT_EQ(-1,a_det.val_.val());
   EXPECT_FLOAT_EQ(1,a_det.d_.val());

  EXPECT_THROW(determinant(matrix_fvv(2,3)), std::domain_error);
}
