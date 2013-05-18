#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/hypot.hpp>

TEST(AgradFvar, hypot) {
  using stan::agrad::fvar;
  using boost::math::hypot;
  using std::isnan;

  fvar<double> x(0.5);
  fvar<double> y(2.3);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = hypot(x, y);
  EXPECT_FLOAT_EQ(hypot(0.5, 2.3), a.val_);
  EXPECT_FLOAT_EQ((0.5 * 1.0 + 2.3 * 2.0) / hypot(0.5, 2.3), a.d_);

  fvar<double> z(0.0);
  fvar<double> w(-2.3);
  z.d_ = 1.0;
  w.d_ = 2.0;
  fvar<double> b = hypot(x, z);
  EXPECT_FLOAT_EQ(0.5, b.val_);
  EXPECT_FLOAT_EQ(1.0, b.d_);

  fvar<double> c = hypot(x, w);
  isnan(c.val_);
  isnan(c.d_);

  fvar<double> d = hypot(z, x);
  EXPECT_FLOAT_EQ(0.5, d.val_);
  EXPECT_FLOAT_EQ(1.0, d.d_);
}
