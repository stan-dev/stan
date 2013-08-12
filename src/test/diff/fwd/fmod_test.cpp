#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, fmod) {
  using stan::agrad::fvar;
  using std::fmod;
  using std::floor;

  fvar<double> x(2.0);
  fvar<double> y(3.0);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = fmod(x, y);
  EXPECT_FLOAT_EQ(fmod(2.0, 3.0), a.val_);
  EXPECT_FLOAT_EQ(1.0 * 1.0 + 2.0 * -floor(2.0 / 3.0), a.d_);

  double z = 4.0;
  double w = 3.0;

  fvar<double> e = fmod(x, z);
  EXPECT_FLOAT_EQ(fmod(2.0, 4.0), e.val_);
  EXPECT_FLOAT_EQ(1.0 / 4.0, e.d_);

  fvar<double> f = fmod(w, x);
  EXPECT_FLOAT_EQ(fmod(3.0, 2.0), f.val_);
  EXPECT_FLOAT_EQ(1.0 * -floor(3.0 / 2.0), f.d_);
 }
