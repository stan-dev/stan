#include <limits>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdAbs,Fvar) {
  using stan::agrad::fvar;
  using std::abs;
  using std::isnan;

  fvar<double> x(2);
  fvar<double> y(-3);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = abs(x);
  EXPECT_FLOAT_EQ(abs(2), a.val_);
  EXPECT_FLOAT_EQ(1.0, a.d_);

  fvar<double> b = abs(-x);
  EXPECT_FLOAT_EQ(abs(-2), b.val_);
  EXPECT_FLOAT_EQ(1.0, b.d_);

  fvar<double> c = abs(y);
  EXPECT_FLOAT_EQ(abs(-3), c.val_);
  EXPECT_FLOAT_EQ(-2.0, c.d_);

  fvar<double> d = abs(2 * x);
  EXPECT_FLOAT_EQ(abs(2 * 2), d.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, d.d_);

  fvar<double> e = abs(y + 4);
  EXPECT_FLOAT_EQ(abs(-3 + 4), e.val_);
  EXPECT_FLOAT_EQ(2.0, e.d_);

  fvar<double> f = abs(x - 2);
  EXPECT_FLOAT_EQ(abs(2 - 2), f.val_);
  EXPECT_FLOAT_EQ(0, f.d_);

  fvar<double> z = std::numeric_limits<double>::quiet_NaN();
  fvar<double> g = abs(z);
  EXPECT_TRUE(boost::math::isnan(g.val_));
  EXPECT_TRUE(boost::math::isnan(g.d_));

  fvar<double> w = 0;
  fvar<double> h = abs(w);
  EXPECT_FLOAT_EQ(0.0, h.val_);
  EXPECT_FLOAT_EQ(0.0, h.d_);
}
TEST(AgradFwdAbs,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::abs;

  fvar<var> x(2.0,1.0);
  fvar<var> a = abs(x);

  EXPECT_FLOAT_EQ(2.0, a.val_.val());
  EXPECT_FLOAT_EQ(1.0, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}
TEST(AgradFwdAbs,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::abs;

  fvar<var> x(2.0,1.0);
  fvar<var> a = abs(x);

  AVEC z = createAVEC(x.val_);
  VEC h;
  a.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0, h[0]);
}
TEST(AgradFwdAbs,FvarFvarDouble) {
  using stan::agrad::fvar;
  using std::abs;

  fvar<fvar<double> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = abs(x);

  EXPECT_FLOAT_EQ(4.0, a.val_.val_);
  EXPECT_FLOAT_EQ(1.0, a.val_.d_);
  EXPECT_FLOAT_EQ(0.0, a.d_.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_.d_);
}
TEST(AgradFwdAbs,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::abs;

  fvar<fvar<var> > y;
  y.val_ = fvar<var>(4.0,1.0);

  fvar<fvar<var> > b = abs(y);

  EXPECT_FLOAT_EQ(4.0, b.val_.val_.val());
  EXPECT_FLOAT_EQ(1.0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(0.0, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0.0, b.d_.d_.val());

  AVEC z = createAVEC(y.val_.val_);
  VEC h;
  b.val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(1.0, h[0]);
}
TEST(AgradFwdAbs,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::abs;

  fvar<fvar<var> > y;
  y.val_ = fvar<var>(4.0,1.0);

  fvar<fvar<var> > b = abs(y);

  AVEC z = createAVEC(y.val_.val_);
  VEC h;
  b.val_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0, h[0]);
}
TEST(AgradFwdAbs,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::abs;

  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.val_.d_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = abs(y);

  AVEC z = createAVEC(y.val_.val_);
  VEC h;
  b.d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0, h[0]);
}

struct abs_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return abs(arg1);
  }
};

TEST(AgradFwdAbs,abs_NaN) {
  abs_fun abs_;
  test_nan(abs_,false);
}
