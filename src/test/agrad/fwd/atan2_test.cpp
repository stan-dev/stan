#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, atan2) {
  using stan::agrad::fvar;
  using std::atan2;

  fvar<double> x(0.5,1.0);
  fvar<double> y(2.3,2.0);
  double w = 2.1;

  fvar<double> a = atan2(x, y);
  EXPECT_FLOAT_EQ(atan2(0.5, 2.3), a.val_);
  EXPECT_FLOAT_EQ((1.0 * 2.3 - 0.5 * 2.0) / (0.5 * 0.5 + 2.3 * 2.3), a.d_);

  fvar<double> b = atan2(w, x);
  EXPECT_FLOAT_EQ(atan2(2.1, 0.5), b.val_);
  EXPECT_FLOAT_EQ((-2.1 * 1.0) / (2.1 * 2.1 + 0.5 * 0.5), b.d_);

  fvar<double> c = atan2(x, w);
  EXPECT_FLOAT_EQ(atan2(0.5, 2.1), c.val_);
  EXPECT_FLOAT_EQ((1.0 * 2.1) / (0.5 * 0.5 + 2.1 * 2.1), c.d_);
}

TEST(AgradFvarVar, atan2) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::atan2;

  fvar<var> x(1.5,1.3);
  fvar<var> z(1.5,1.0);
  fvar<var> a = atan2(x,z);

  EXPECT_FLOAT_EQ(atan2(1.5,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(0.1, a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / 3.0, g[0]);
  EXPECT_FLOAT_EQ(-1.0 / 3.0,g[1]);
}

TEST(AgradFvarFvar, atan2) {
  using stan::agrad::fvar;
  using std::atan2;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = atan2(x,y);

  EXPECT_FLOAT_EQ(atan(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(-1.5 / (1.5 * 1.5 + 1.5 * 1.5), a.d_.val_);
  EXPECT_FLOAT_EQ((1.5 * 1.5 - 1.5 * 1.5) / ((1.5 * 1.5 + 1.5 * 1.5) * (1.5 * 1.5 + 1.5 * 1.5)), a.d_.d_);
}
