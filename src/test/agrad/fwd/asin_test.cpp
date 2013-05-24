#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>
#include <stan/math/constants.hpp>

TEST(AgradFvar, asin) {
  using stan::agrad::fvar;
  using std::asin;
  using std::isnan;
  using std::sqrt;
  using stan::math::INFTY;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = asin(x);
  EXPECT_FLOAT_EQ(asin(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(1 - 0.5 * 0.5), a.d_);

  fvar<double> b = 2 * asin(x) + 4;
  EXPECT_FLOAT_EQ(2 * asin(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / sqrt(1 - 0.5 * 0.5), b.d_);

  fvar<double> c = -asin(x) + 5;
  EXPECT_FLOAT_EQ(-asin(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / sqrt(1 - 0.5 * 0.5), c.d_);

  fvar<double> d = -3 * asin(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * asin(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / sqrt(1 - 0.5 * 0.5) + 5, d.d_);

  fvar<double> y(3.4);
  y.d_ = 1.0;
  fvar<double> e = asin(y);
  isnan(e.val_);
  isnan(e.d_);

  fvar<double> z(1.0);
  z.d_ = 1.0;
  fvar<double> f = asin(z);
  EXPECT_FLOAT_EQ(asin(1.0), f.val_);
  EXPECT_FLOAT_EQ(INFTY, f.d_);
}

TEST(AgradFvarVar, asin) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::asin;

  fvar<var> x;
  x.val_ = 0.5;
  x.d_ = 0.3;
  fvar<var> a = asin(x);

  EXPECT_FLOAT_EQ(asin(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(0.3 / sqrt(1.0 - 0.5 * 0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / sqrt(1.0 - 0.5 * 0.5), g[0]);

  y = createAVEC(x.d_);
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFvarFvar, asin) {
  using stan::agrad::fvar;
  using std::asin;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 2.0;
  x.d_.val_ = 3.0;
  x.d_.d_ = 4.0;

  fvar<fvar<double> > a = asin(x);

  EXPECT_FLOAT_EQ(asin(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 - 0.5 * 0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(3.0 / sqrt(1.0 - 0.5 * 0.5), a.d_.val_);
  EXPECT_FLOAT_EQ(8.0 / sqrt(1.0 - 0.5 * 0.5), a.d_.d_);
}
