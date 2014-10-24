#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdOperatorDivision, Fvar) {
  using stan::agrad::fvar;
  using std::isnan;

  fvar<double> x1(0.5,1.0);
  fvar<double> x2(0.4,2.0);
  fvar<double> a = x1 / x2;

  EXPECT_FLOAT_EQ(0.5 / 0.4, a.val_);
  EXPECT_FLOAT_EQ((1.0 * 0.4 - 2.0 * 0.5) / (0.4 * 0.4), a.d_);

  fvar<double> b = -x1 / x2;
  EXPECT_FLOAT_EQ(-0.5 / 0.4, b.val_);
  EXPECT_FLOAT_EQ((-1 * 0.4 + 2.0 * 0.5) / (0.4 * 0.4), b.d_);

  fvar<double> c = -3 * x1 / x2;
  EXPECT_FLOAT_EQ(-3 * 0.5 / 0.4, c.val_);
  EXPECT_FLOAT_EQ(3 * (-1 * 0.4 + 2.0 * 0.5) / (0.4 * 0.4), c.d_);

  fvar<double> x3(0.5,1.0);
  double x4 = 2.0;

  fvar<double> e = x4 / x3;
  EXPECT_FLOAT_EQ(2 / 0.5, e.val_);
  EXPECT_FLOAT_EQ(-2 * 1.0 / (0.5 * 0.5), e.d_);

  fvar<double> f = x3 / -2;
  EXPECT_FLOAT_EQ(0.5 / -2, f.val_);
  EXPECT_FLOAT_EQ(1.0 / -2, f.d_);

  fvar<double> x5(0.0,1.0);
  fvar<double> g = x3/x5;
  isnan(g.val_);
  isnan(g.d_);
}

TEST(AgradFwdOperatorDivision, FvarVar_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  fvar<var> z(0.5,1.3);
  fvar<var> a = x / z;

  EXPECT_FLOAT_EQ(1.0, a.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(2, g[0]);
  EXPECT_FLOAT_EQ(-0.5 / 0.25, g[1]);
}
TEST(AgradFwdOperatorDivision, FvarVar_Double_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  double z(0.5);
  fvar<var> a = x / z;

  EXPECT_FLOAT_EQ(1.0, a.val_.val());
  EXPECT_FLOAT_EQ(2.6, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(2, g[0]);
}
TEST(AgradFwdOperatorDivision, Double_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  double x(0.5);
  fvar<var> z(0.5,1.3);
  fvar<var> a = x / z;

  EXPECT_FLOAT_EQ(1.0, a.val_.val());
  EXPECT_FLOAT_EQ(-2.6, a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.5 / 0.25, g[0]);
}
TEST(AgradFwdOperatorDivision, FvarVar_FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  fvar<var> z(0.5,1.3);
  fvar<var> a = x / z;

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-5.2, g[0]);
  EXPECT_FLOAT_EQ(5.2, g[1]);
}
TEST(AgradFwdOperatorDivision, FvarVar_Double_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  double z(0.5);
  fvar<var> a = x / z;

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdOperatorDivision, Double_FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  double x(0.5);
  fvar<var> z(0.5,1.3);
  fvar<var> a = x / z;

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(10.4, g[0]);
}

TEST(AgradFwdOperatorDivision, FvarFvarDouble) {
  using stan::agrad::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > z = x / y;
  EXPECT_FLOAT_EQ(1, z.val_.val_);
  EXPECT_FLOAT_EQ(1 / 0.5, z.val_.d_);
  EXPECT_FLOAT_EQ(-0.5 / 0.25 , z.d_.val_);
  EXPECT_FLOAT_EQ(-1.0 / 0.25, z.d_.d_);
}
TEST(AgradFwdOperatorDivision, FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z = x / y;
  EXPECT_FLOAT_EQ(1, z.val_.val_.val());
  EXPECT_FLOAT_EQ(1 / 0.5, z.val_.d_.val());
  EXPECT_FLOAT_EQ(-0.5 / 0.25 , z.d_.val_.val());
  EXPECT_FLOAT_EQ(-1.0 / 0.25, z.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  z.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0 / 0.5, g[0]);
  EXPECT_FLOAT_EQ(-0.5 / 0.25, g[1]);
}
TEST(AgradFwdOperatorDivision, FvarFvarVar_Double_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  double y(0.5);

  fvar<fvar<var> > z = x / y;
  EXPECT_FLOAT_EQ(1, z.val_.val_.val());
  EXPECT_FLOAT_EQ(1 / 0.5, z.val_.d_.val());
  EXPECT_FLOAT_EQ(0 , z.d_.val_.val());
  EXPECT_FLOAT_EQ(0, z.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  z.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0 / 0.5, g[0]);
}

TEST(AgradFwdOperatorDivision, Double_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  double x(0.5);
  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z = x / y;
  EXPECT_FLOAT_EQ(1, z.val_.val_.val());
  EXPECT_FLOAT_EQ(0, z.val_.d_.val());
  EXPECT_FLOAT_EQ(-0.5 / 0.25 , z.d_.val_.val());
  EXPECT_FLOAT_EQ(0, z.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  z.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.5 / 0.25, g[0]);
}
TEST(AgradFwdOperatorDivision, FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z = x / y;

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  z.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(-1.0 / 0.25, g[1]);
}
TEST(AgradFwdOperatorDivision, FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z = x / y;

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  z.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-1.0 / 0.25, g[0]);
  EXPECT_FLOAT_EQ(8, g[1]);
}
TEST(AgradFwdOperatorDivision, FvarFvarVar_Double_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  double y(0.5);

  fvar<fvar<var> > z = x / y;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  z.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFwdOperatorDivision, Double_FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  double x(0.5);
  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z = x / y;

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  z.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(8, g[0]);
}
TEST(AgradFwdOperatorDivision, FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z = x / y;

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  z.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(16, g[1]);
}
TEST(AgradFwdOperatorDivision, FvarFvarVar_Double_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  double y(0.5);

  fvar<fvar<var> > z = x / y;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  z.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFwdOperatorDivision, Double_FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  double x(0.5);
  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > z = x / y;

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  z.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-48, g[0]);
}

struct divide_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return arg1/arg2;
  }
};

TEST(AgradFwdOperatorDivision, divide_nan) {
  divide_fun divide_;
  test_nan(divide_,3.0,5.0,false);
}
