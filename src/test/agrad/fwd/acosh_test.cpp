#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/acosh.hpp>

TEST(AgradFvar, acosh) {
  using stan::agrad::fvar;
  using boost::math::acosh;
  using std::sqrt;
  using std::isnan;

  fvar<double> x(1.5);
  x.d_ = 1.0;

  fvar<double> a = acosh(x);
  EXPECT_FLOAT_EQ(acosh(1.5), a.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(-1 + (1.5) * (1.5)), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = acosh(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> z(0.5);
  z.d_ = 1.0;

  fvar<double> c = acosh(z);
  isnan(c.val_);
  isnan(c.d_);
}
