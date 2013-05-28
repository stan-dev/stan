#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

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

TEST(AgradFvarVar, fmod) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::fmod;

  fvar<var> x;
  x.val_ = 3.0;
  x.d_ = 1.3;

  fvar<var> z;
  z.val_ = 6.0;
  z.d_ = 1.0;
  fvar<var> a = fmod(x,z);

  EXPECT_FLOAT_EQ(fmod(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ(1.3, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1,g[0]);
  std::isnan(g[1]);

  y = createAVEC(x.d_);
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0,g[0]);
  std::isnan(g[1]);
}

TEST(AgradFvarFvar, fmod) {
  using stan::agrad::fvar;
  using std::fmod;

  fvar<fvar<double> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  x.d_.val_ = 0.0;
  x.d_.d_ = 0.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.val_.d_ = 0.0;
  y.d_.val_ = 1.0;
  y.d_.d_ = 0.0;

  fvar<fvar<double> > a = fmod(x,y);

  EXPECT_FLOAT_EQ(fmod(3.0,6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(1, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
