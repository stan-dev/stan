#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, pow) {
  using stan::agrad::fvar;
  using std::pow;
  using std::log;
  using std::isnan;

  fvar<double> x(0.5);
  x.d_ = 1.0;
  double y = 5.0;

  fvar<double> a = pow(x, y);
  EXPECT_FLOAT_EQ(pow(0.5, 5.0), a.val_);
  EXPECT_FLOAT_EQ(5.0 * pow(0.5, 5.0 - 1.0), a.d_);

  fvar<double> b = pow(y, x);
  EXPECT_FLOAT_EQ(pow(5.0, 0.5), b.val_);
  EXPECT_FLOAT_EQ(log(5.0) * pow(5.0, 0.5), b.d_);

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

TEST(AgradFvarVar, pow) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::log;
  using std::pow;

  fvar<var> x;
  x.val_ = 0.5;
  x.d_ = 1.0;

  fvar<var> z;
  z.val_ = 1.2;
  z.d_ = 2.0;
  fvar<var> a = pow(x,z);

  EXPECT_FLOAT_EQ(pow(0.5,1.2), a.val_.val());
  EXPECT_FLOAT_EQ((2.0 * log(0.5) + 1.2 * 1.0 / 0.5) * pow(0.5, 1.2), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.2 / 0.5 * pow(0.5, 1.2), g[0]);
  std::isnan(g[1]);

  y = createAVEC(x.d_);
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  std::isnan(g[1]);
}

TEST(AgradFvarFvar, pow) {
  using stan::agrad::fvar;
  using std::pow;
  using std::log;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 0.0;
  x.d_.d_ = 0.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.val_.d_ = 0.0;
  y.d_.val_ = 1.0;
  y.d_.d_ = 0.0;

  fvar<fvar<double> > a = pow(x,y);

  EXPECT_FLOAT_EQ(pow(0.5,0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0.5 * pow(0.5,-0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(log(0.5) * pow(0.5,0.5), a.d_.val_);
  EXPECT_FLOAT_EQ(pow(0.5, -0.5) * (0.5 * log(0.5) + 1.0), a.d_.d_);
}
