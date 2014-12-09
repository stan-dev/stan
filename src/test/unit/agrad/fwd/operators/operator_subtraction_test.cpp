#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdOperatorSubtraction, Fvar) {
  using stan::agrad::fvar;

  fvar<double> x1(0.5,1.0);
  fvar<double> x2(0.4,2.0);

  fvar<double> a = x1 - x2;
  EXPECT_FLOAT_EQ(0.5 - 0.4, a.val_);
  EXPECT_FLOAT_EQ(1.0 - 2.0, a.d_);

  fvar<double> b = -x1 - x2;
  EXPECT_FLOAT_EQ(-0.5 - 0.4, b.val_);
  EXPECT_FLOAT_EQ(-1 * 1.0 - 2.0, b.d_);

  fvar<double> c = 2 * x1 - -3 * x2;
  EXPECT_FLOAT_EQ(2 * 0.5 - -3 * 0.4, c.val_);
  EXPECT_FLOAT_EQ(2 * 1.0 - -3 * 2.0, c.d_);

  fvar<double> x3(0.5,1.0);
  fvar<double> x4(1.0,2.0);

  fvar<double> d = 2 * x3 - x4;
  EXPECT_FLOAT_EQ(2 * 0.5 - 1 * 1.0, d.val_);
  EXPECT_FLOAT_EQ(2 * 1.0 - 1 * 2.0, d.d_);

  fvar<double> e = 2 * x3 - 4;
  EXPECT_FLOAT_EQ(2 * 0.5 - 4, e.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, e.d_);

  fvar<double> f = 5 - 2 * x3;
  EXPECT_FLOAT_EQ(5 - 2 * 0.5, f.val_);
  EXPECT_FLOAT_EQ(-2 * 1.0, f.d_);
}

TEST(AgradFwdOperatorSubtraction, FvarVar_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  fvar<var> z(0.5,1.3);
  fvar<var> a = x - z;

  EXPECT_FLOAT_EQ(0.0, a.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1, g[0]);
  EXPECT_FLOAT_EQ(-1, g[1]);
}
TEST(AgradFwdOperatorSubtraction, FvarVar_Double_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  double z(0.5);
  fvar<var> a = x - z;

  EXPECT_FLOAT_EQ(0.0, a.val_.val());
  EXPECT_FLOAT_EQ(1.3, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1, g[0]);
}
TEST(AgradFwdOperatorSubtraction, Double_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  double x(0.5);
  fvar<var> z(0.5,1.3);
  fvar<var> a = x - z;

  EXPECT_FLOAT_EQ(0.0, a.val_.val());
  EXPECT_FLOAT_EQ(-1.3, a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-1, g[0]);
}
TEST(AgradFwdOperatorSubtraction, FvarVar_FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  fvar<var> z(0.5,1.3);
  fvar<var> a = x - z;

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(0, g[1]);
}
TEST(AgradFwdOperatorSubtraction, FvarVar_Double_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  double z(0.5);
  fvar<var> a = x - z;

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdOperatorSubtraction, Double_FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  double x(0.5);
  fvar<var> z(0.5,1.3);
  fvar<var> a = x - z;

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFwdOperatorSubtraction, FvarFvarDouble) {
  using stan::agrad::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > z = x - y;
  EXPECT_FLOAT_EQ(0, z.val_.val_);
  EXPECT_FLOAT_EQ(1, z.val_.d_);
  EXPECT_FLOAT_EQ(-1, z.d_.val_);
  EXPECT_FLOAT_EQ(0, z.d_.d_);
}
TEST(AgradFwdOperatorSubtraction, FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z = x - y;
  EXPECT_FLOAT_EQ(0, z.val_.val_.val());
  EXPECT_FLOAT_EQ(1, z.val_.d_.val());
  EXPECT_FLOAT_EQ(-1, z.d_.val_.val());
  EXPECT_FLOAT_EQ(0, z.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  z.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
  EXPECT_FLOAT_EQ(-1.0, g[1]);
}
TEST(AgradFwdOperatorSubtraction, FvarFvarVar_Double_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  double y(0.5);

  fvar<fvar<var> > z = x - y;
  EXPECT_FLOAT_EQ(0, z.val_.val_.val());
  EXPECT_FLOAT_EQ(1, z.val_.d_.val());
  EXPECT_FLOAT_EQ(0, z.d_.val_.val());
  EXPECT_FLOAT_EQ(0, z.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  z.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}
TEST(AgradFwdOperatorSubtraction, Double_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  double x(0.5);
  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z = x - y;
  EXPECT_FLOAT_EQ(0, z.val_.val_.val());
  EXPECT_FLOAT_EQ(0, z.val_.d_.val());
  EXPECT_FLOAT_EQ(-1, z.d_.val_.val());
  EXPECT_FLOAT_EQ(0, z.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  z.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-1.0, g[0]);
}

TEST(AgradFwdOperatorSubtraction, FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z = x - y;

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  z.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(0, g[1]);
}
TEST(AgradFwdOperatorSubtraction, FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z = x - y;

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  z.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(0, g[1]);
}
TEST(AgradFwdOperatorSubtraction, FvarFvarVar_Double_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  double y(0.5);

  fvar<fvar<var> > z = x - y;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  z.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdOperatorSubtraction, Double_FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  double x(0.5);
  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z = x - y;

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  z.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdOperatorSubtraction, FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > z = x - y;

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  z.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(0, g[1]);
}
TEST(AgradFwdOperatorSubtraction, FvarFvarVar_Double_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;
  double y(0.5);

  fvar<fvar<var> > z = x - y;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  z.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdOperatorSubtraction, Double_FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  double x(0.5);
  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > z = x - y;

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  z.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

struct subtract_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return arg1-arg2;
  }
};

TEST(AgradFwdOperatorSubtraction, subtract_nan) {
  subtract_fun subtract_;
  test_nan(subtract_,3.0,5.0,false);
}
