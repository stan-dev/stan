#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>

TEST(AgradMixOperatorDivision, FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixOperatorDivision, FvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixOperatorDivision, Double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixOperatorDivision, FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(0.5,1.3);
  fvar<var> z(0.5,1.3);
  fvar<var> a = x / z;

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-5.2, g[0]);
  EXPECT_FLOAT_EQ(5.2, g[1]);
}
TEST(AgradMixOperatorDivision, FvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(0.5,1.3);
  double z(0.5);
  fvar<var> a = x / z;

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradMixOperatorDivision, Double_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(0.5);
  fvar<var> z(0.5,1.3);
  fvar<var> a = x / z;

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(10.4, g[0]);
}

TEST(AgradMixOperatorDivision, FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixOperatorDivision, FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

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

TEST(AgradMixOperatorDivision, Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixOperatorDivision, FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixOperatorDivision, FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixOperatorDivision, FvarFvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

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

TEST(AgradMixOperatorDivision, Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixOperatorDivision, FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

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
TEST(AgradMixOperatorDivision, FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

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

TEST(AgradMixOperatorDivision, Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

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

TEST(AgradMixOperatorDivision, divide_nan) {
  divide_fun divide_;
  test_nan_mix(divide_,3.0,5.0,false);
}
