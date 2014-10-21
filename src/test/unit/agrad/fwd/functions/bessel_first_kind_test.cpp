#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdBesselFirstKind,Fvar) {
  using stan::agrad::fvar;
  using stan::agrad::bessel_first_kind;

  fvar<double> a(4.0,1.0);
  int b = 0;
  fvar<double> x = bessel_first_kind(b,a);
  EXPECT_FLOAT_EQ(-0.397149809863847372, x.val_);
  EXPECT_FLOAT_EQ(0.0660433280235491361, x.d_);

  fvar<double> c(-3.0,2.0);

  x = bessel_first_kind(1, c);
  EXPECT_FLOAT_EQ(-0.3390589585259364589255, x.val_);
  EXPECT_FLOAT_EQ(-0.7461432154878245145319, x.d_);
}

TEST(AgradFwdBesselFirstKind,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::bessel_first_kind;

  fvar<var> z(-3.0,2.0);
  fvar<var> a = bessel_first_kind(1,z);

  EXPECT_FLOAT_EQ(bessel_first_kind(1, -3.0), a.val_.val());
  EXPECT_FLOAT_EQ(-0.7461432154878245145319, a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.7461432154878245145319 / 2.0, g[0]);
}
TEST(AgradFwdBesselFirstKind,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::bessel_first_kind;

  fvar<var> z(-3.0,2.0);
  fvar<var> a = bessel_first_kind(1,z);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.35405707, g[0]);
}
TEST(AgradFwdBesselFirstKind,FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::bessel_first_kind;

  fvar<fvar<double> > x;
  x.val_.val_ = -3.0;
  x.val_.d_ = 2.0;
  fvar<fvar<double> > y;
  y.val_.val_ = -3.0;
  y.d_.val_ = 2.0;

  fvar<fvar<double> > a = bessel_first_kind(1,y);

  EXPECT_FLOAT_EQ(bessel_first_kind(1,-3.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(-0.7461432154878245145319, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > b = bessel_first_kind(1, x);

  EXPECT_FLOAT_EQ(bessel_first_kind(1,-3.0), b.val_.val_);
  EXPECT_FLOAT_EQ(-0.7461432154878245145319, b.val_.d_);
  EXPECT_FLOAT_EQ(0, b.d_.val_);
  EXPECT_FLOAT_EQ(0, b.d_.d_);
}
TEST(AgradFwdBesselFirstKind,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::bessel_first_kind;

  fvar<fvar<var> > y;
  y.val_.val_ = -3.0;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > a = bessel_first_kind(1,y);

  EXPECT_FLOAT_EQ(bessel_first_kind(1,-3.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(-0.7461432154878245145319, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.7461432154878245145319 / 2.0, g[0]);

  fvar<fvar<var> > x;
  x.val_.val_ = -3.0;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > b = bessel_first_kind(1, x);

  EXPECT_FLOAT_EQ(bessel_first_kind(1,-3.0), b.val_.val_.val());
  EXPECT_FLOAT_EQ(-0.7461432154878245145319, b.val_.d_.val());
  EXPECT_FLOAT_EQ(0, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-0.7461432154878245145319 / 2.0, r[0]);
}
TEST(AgradFwdBesselFirstKind,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::bessel_first_kind;

  fvar<fvar<var> > y;
  y.val_.val_ = -3.0;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > a = bessel_first_kind(1,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.35405707, g[0]);

  fvar<fvar<var> > x;
  x.val_.val_ = -3.0;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > b = bessel_first_kind(1, x);

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  b.val_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0.35405707, r[0]);
}
TEST(AgradFwdBesselFirstKind,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::bessel_first_kind;

  fvar<fvar<var> > y;
  y.val_.val_ = -3.0;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = bessel_first_kind(1,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.32406084039059405127, g[0]);
}

struct bessel_first_kind_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return bessel_first_kind(1,arg1);
  }
};

TEST(AgradFwdBesselFirstKind,bessel_first_kind_NaN) {
  bessel_first_kind_fun bessel_first_kind_;
  test_nan(bessel_first_kind_,true);
}
