#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdTan, Fvar) {
  using stan::agrad::fvar;
  using std::tan;
  using std::cos;

  fvar<double> x(0.5,1.0);
  fvar<double> a = tan(x);
  EXPECT_FLOAT_EQ(tan(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (cos(0.5) * cos(0.5)), a.d_);

  fvar<double> b = 2 * tan(x) + 4;
  EXPECT_FLOAT_EQ(2 * tan(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / (cos(0.5) * cos(0.5)), b.d_);

  fvar<double> c = -tan(x) + 5;
  EXPECT_FLOAT_EQ(-tan(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (cos(0.5) * cos(0.5)), c.d_);

  fvar<double> d = -3 * tan(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * tan(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (cos(0.5) * cos(0.5)) + 5, d.d_);

  fvar<double> y(-0.5,1.0);
  fvar<double> e = tan(y);
  EXPECT_FLOAT_EQ(tan(-0.5), e.val_);
  EXPECT_FLOAT_EQ(1 / (cos(-0.5) * cos(-0.5)), e.d_);

  fvar<double> z(0.0,1.0);
  fvar<double> f = tan(z);
  EXPECT_FLOAT_EQ(tan(0.0), f.val_);
  EXPECT_FLOAT_EQ(1 / (cos(0.0) * cos(0.0)), f.d_);
}

TEST(AgradFwdTan, FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::tan;
  using std::cos;

  fvar<var> x(1.5,1.3);
  fvar<var> a = tan(x);

  EXPECT_FLOAT_EQ(tan(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / (cos(1.5) * cos(1.5)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / (cos(1.5) * cos(1.5)), g[0]);
}
TEST(AgradFwdTan, FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::tan;
  using std::cos;

  fvar<var> x(1.5,1.3);
  fvar<var> a = tan(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(2.0 / (cos(1.5) * cos(1.5)) * tan(1.5) * 1.3, g[0]);
}
TEST(AgradFwdTan, FvarFvarDouble) {
  using stan::agrad::fvar;
  using std::tan;
  using std::cos;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = tan(x);

  EXPECT_FLOAT_EQ(tan(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 / (cos(1.5) * cos(1.5)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = tan(y);
  EXPECT_FLOAT_EQ(tan(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 / (cos(1.5) * cos(1.5)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdTan, FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::tan;
  using std::cos;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = tan(x);

  EXPECT_FLOAT_EQ(tan(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(2.0 / (cos(1.5) * cos(1.5)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0 / (cos(1.5) * cos(1.5)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = tan(y);
  EXPECT_FLOAT_EQ(tan(1.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(2.0 / (cos(1.5) * cos(1.5)), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.0 / (cos(1.5) * cos(1.5)), r[0]);
}
TEST(AgradFwdTan, FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::tan;
  using std::cos;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = tan(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(2.0 * 2.0 * tan(1.5) / (cos(1.5) * cos(1.5)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = tan(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.0 * 2.0 * tan(1.5) / (cos(1.5) * cos(1.5)), r[0]);
}
TEST(AgradFwdTan, FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::tan;
  using std::cos;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = tan(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(238840.84160534013669260979995, g[0]);
}

struct tan_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return tan(arg1);
  }
};

TEST(AgradFwdTan,tan_NaN) {
  tan_fun tan_;
  test_nan(tan_,false);
}
