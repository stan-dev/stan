#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, pow) {
  using stan::agrad::fvar;
  using std::pow;
  using std::log;
  using std::isnan;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> z(1.2);
  z.d_ = 2.0;
  fvar<double> c = pow(x, z);
  EXPECT_FLOAT_EQ(pow(0.5, 1.2), c.val_);
  EXPECT_FLOAT_EQ((2.0 * log(0.5) + 1.2 * 1.0 / 0.5) * pow(0.5, 1.2), c.d_);

  fvar<double> w(-0.4);
  w.d_ = 1.0;
  fvar<double> d = pow(w, x);
  isnan(d.val_);
  isnan(d.d_);
}
