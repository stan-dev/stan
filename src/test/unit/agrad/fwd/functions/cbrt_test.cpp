#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <boost/math/special_functions/cbrt.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdCbrt,Fvar) {
  using stan::agrad::fvar;
  using boost::math::cbrt;
  using std::isnan;

  fvar<double> x(0.5,1.0);
  fvar<double> a = cbrt(x);

  EXPECT_FLOAT_EQ(cbrt(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (3 * pow(0.5, 2.0 / 3.0)), a.d_);

  fvar<double> b = 3 * cbrt(x) + x;
  EXPECT_FLOAT_EQ(3 * cbrt(0.5) + 0.5, b.val_);
  EXPECT_FLOAT_EQ(3 / (3 * pow(0.5, 2.0 / 3.0)) + 1, b.d_);

  fvar<double> c = -cbrt(x) + 5;
  EXPECT_FLOAT_EQ(-cbrt(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (3 * pow(0.5, 2.0 / 3.0)), c.d_);

  fvar<double> d = -3 * cbrt(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * cbrt(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (3 * pow(0.5, 2.0 / 3.0)) + 5, d.d_);

  fvar<double> e = -3 * cbrt(-x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * cbrt(-0.5) + 5 * 0.5, e.val_);
  EXPECT_FLOAT_EQ(3 / (3 * cbrt(-0.5) * cbrt(-0.5)) + 5, e.d_);

  fvar<double> y(0.0,1.0);
  fvar<double> f = cbrt(y);
  EXPECT_FLOAT_EQ(cbrt(0.0), f.val_);
  isnan(f.d_);
}

TEST(AgradFwdCbrt,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::cbrt;

  fvar<var> x(1.5,1.3);
  fvar<var> a = cbrt(x);

  EXPECT_FLOAT_EQ(cbrt(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / (3 * cbrt(1.5) * cbrt(1.5)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / (3.0 * cbrt(1.5) * cbrt(1.5)), g[0]);
}

TEST(AgradFwdCbrt,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::cbrt;

  fvar<var> x(1.5,1.3);
  fvar<var> a = cbrt(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-2.0 * 1.3 / 3.0 / (3.0 * cbrt(1.5) * cbrt(1.5) * 1.5), g[0]);
}

TEST(AgradFwdCbrt,FvarFvarDouble) {
  using stan::agrad::fvar;
  using boost::math::cbrt;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = cbrt(x);

  EXPECT_FLOAT_EQ(cbrt(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 / (3.0 * cbrt(1.5) * cbrt(1.5)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = cbrt(y);
  EXPECT_FLOAT_EQ(cbrt(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 / (3.0 * cbrt(1.5) * cbrt(1.5)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

TEST(AgradFwdCbrt,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::cbrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = cbrt(x);

  EXPECT_FLOAT_EQ(cbrt(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(2.0 / (3.0 * cbrt(1.5) * cbrt(1.5)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0 / (3.0 * cbrt(1.5) * cbrt(1.5)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = cbrt(y);
  EXPECT_FLOAT_EQ(cbrt(1.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(2.0 / (3.0 * cbrt(1.5) * cbrt(1.5)), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());


  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.0 / (3.0 * cbrt(1.5) * cbrt(1.5)), r[0]);
}

TEST(AgradFwdCbrt,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::cbrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = cbrt(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(2.0 * -2.0 / 3.0 / (3.0 * cbrt(1.5) * cbrt(1.5) * 1.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = cbrt(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.0 * -2.0 / 3.0 / (3.0 * cbrt(1.5) * cbrt(1.5) * 1.5), r[0]);
}
TEST(AgradFwdCbrt,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::cbrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = cbrt(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.12562021866154533528757664877253, g[0]);
}

struct cbrt_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return cbrt(arg1);
  }
};

TEST(AgradFwdCbrt,cbrt_NaN) {
  cbrt_fun cbrt_;
  test_nan(cbrt_,false);
}
