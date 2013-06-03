#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/inv_sqrt.hpp>
#include <stan/math/constants.hpp>

TEST(AgradFvar, inv_sqrt) {
  using stan::agrad::fvar;
  using stan::math::inv_sqrt;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  fvar<double> a = inv_sqrt(x);

  EXPECT_FLOAT_EQ(inv_sqrt(0.5), a.val_);
  EXPECT_FLOAT_EQ(-0.5 / (0.5 * std::sqrt(0.5)), a.d_);

  fvar<double> y(0.0);
  y.d_ = 1.0;
  fvar<double> g = inv_sqrt(y);

  EXPECT_FLOAT_EQ(stan::math::positive_infinity(), g.val_);
  EXPECT_FLOAT_EQ(stan::math::negative_infinity(), g.d_);

  fvar<double> z(-1.0);
  z.d_ = 2.0;
  g = inv_sqrt(z);

  std::isnan(g.val_);
  std::isnan(g.d_);
}   
