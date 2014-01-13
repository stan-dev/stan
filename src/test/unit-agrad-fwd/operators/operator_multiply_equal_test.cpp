#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdOperatorMultiplyEqual, Fvar) {
  using stan::agrad::fvar;

  fvar<double> a(0.5,1.0);
  fvar<double> x1(0.4,2.0);
  a *= x1;
  EXPECT_FLOAT_EQ(0.5 * 0.4, a.val_);
  EXPECT_FLOAT_EQ(1.0 * 0.4 + 2.0 * 0.5, a.d_);

  fvar<double> b(0.5,1.0);
  fvar<double> x2(0.4,2.0);
  b *= -x2;
  EXPECT_FLOAT_EQ(0.5 * -0.4, b.val_);
  EXPECT_FLOAT_EQ(1.0 * -0.4 + -2.0 * 0.5, b.d_);

  fvar<double> c(0.6,3.0);
  double x3(0.3);
  c *= x3;
  EXPECT_FLOAT_EQ(0.6 * 0.3, c.val_);
  EXPECT_FLOAT_EQ(3.0, c.d_);

  fvar<double> d(0.5,1.0);
  fvar<double> x4(-0.4,2.0);
  d *= x4;
  EXPECT_FLOAT_EQ(0.5 * -0.4, d.val_);
  EXPECT_FLOAT_EQ(1.0 * -0.4 + 2.0 * 0.5, d.d_);
}

TEST(AgradFwdOperatorMultiplyEqual, FvarVar_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  fvar<var> z(0.5,1.3);
  x *= z;

  EXPECT_FLOAT_EQ(0.25, x.val_.val());
  EXPECT_FLOAT_EQ(1.3, x.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  x.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1, g[0]);
  EXPECT_FLOAT_EQ(0.5, g[1]);
}
TEST(AgradFwdOperatorMultiplyEqual, FvarVar_Double_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  double z(0.5);
  x *= z;

  EXPECT_FLOAT_EQ(0.25, x.val_.val());
  EXPECT_FLOAT_EQ(1.3, x.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  x.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1, g[0]);
}
TEST(AgradFwdOperatorMultiplyEqual, FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  fvar<var> z(0.5,1.3);
  x *= z;

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  x.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(1.3, g[1]);
}
TEST(AgradFwdOperatorMultiplyEqual, FvarVar_Double_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  double z(0.5);
  x *= z;

  AVEC y = createAVEC(x.val_);
  VEC g;
  x.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFwdOperatorMultiplyEqual, FvarFvarDouble) {
  using stan::agrad::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  x *= y;
  EXPECT_FLOAT_EQ(0.25, x.val_.val_);
  EXPECT_FLOAT_EQ(0.5, x.val_.d_);
  EXPECT_FLOAT_EQ(0.5, x.d_.val_);
  EXPECT_FLOAT_EQ(1, x.d_.d_);
}
TEST(AgradFwdOperatorMultiplyEqual, FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  x *= y;
  EXPECT_FLOAT_EQ(0.25, x.val_.val_.val());
  EXPECT_FLOAT_EQ(0.5, x.val_.d_.val());
  EXPECT_FLOAT_EQ(0.5, x.d_.val_.val());
  EXPECT_FLOAT_EQ(1, x.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  x.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
  EXPECT_FLOAT_EQ(0.5, g[1]);
}
TEST(AgradFwdOperatorMultiplyEqual, FvarFvarVar_Double_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  double y(0.5);

  x *= y;
  EXPECT_FLOAT_EQ(0.25, x.val_.val_.val());
  EXPECT_FLOAT_EQ(1, x.val_.d_.val());
  EXPECT_FLOAT_EQ(0, x.d_.val_.val());
  EXPECT_FLOAT_EQ(0, x.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}
TEST(AgradFwdOperatorMultiplyEqual, FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  x *= y;

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  x.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(1, g[1]);
}
TEST(AgradFwdOperatorMultiplyEqual, FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  x *= y;

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  x.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(0, g[1]);
}
TEST(AgradFwdOperatorMultiplyEqual, FvarFvarVar_Double_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  double y(0.5);

  x *= y;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdOperatorMultiplyEqual, FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  x *= y;

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  x.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(0, g[1]);
}
TEST(AgradFwdOperatorMultiplyEqual, FvarFvarVar_Double_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  double y(0.5);

  x *= y;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
