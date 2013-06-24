#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/inv_square.hpp>
#include <stan/math/constants.hpp>

TEST(AgradFvar, inv_square) {
  using stan::agrad::fvar;
  using stan::math::inv_square;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  fvar<double> a = inv_square(x);

  EXPECT_FLOAT_EQ(inv_square(0.5), a.val_);
  EXPECT_FLOAT_EQ(-2 / (0.5 * 0.5 * 0.5), a.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> g = inv_square(z);
  EXPECT_FLOAT_EQ(stan::math::positive_infinity(), g.val_);
  EXPECT_FLOAT_EQ(stan::math::negative_infinity(), g.d_);
}   
