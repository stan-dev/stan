#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/functions/lbeta.hpp>

TEST(AgradFvar, lbeta) {
  using stan::agrad::fvar;
  using boost::math::tgamma;
  using stan::math::lbeta;

  fvar<double> x(0.5);
  fvar<double> y(1.2);
  x.d_ = 1.0;
  y.d_ = 2.0;

  double w = 1.3;

  fvar<double> a = lbeta(x, y);
  EXPECT_FLOAT_EQ(lbeta(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ(1.0 / tgamma(0.5) + 2.0 / tgamma(1.2) 
                  - (1.0 + 2.0) / tgamma(0.5 + 1.2), a.d_);

  fvar<double> b = lbeta(x, w);
  EXPECT_FLOAT_EQ(lbeta(0.5, 1.3), b.val_);
  EXPECT_FLOAT_EQ(1.0 / tgamma(0.5) - 1.0 / tgamma(0.5 + 1.3), b.d_);

  fvar<double> c = lbeta(w, x);
  EXPECT_FLOAT_EQ(lbeta(1.3, 0.5), c.val_);
  EXPECT_FLOAT_EQ(1.0 / tgamma(0.5) - 1.0 / tgamma(1.3 + 0.5), c.d_);
}
