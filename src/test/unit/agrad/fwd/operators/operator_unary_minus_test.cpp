#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdOperatorUnaryMinus, Fvar) {
  using stan::agrad::fvar;

  fvar<double> x1(0.5,1.0);
  fvar<double> a = -x1;
  EXPECT_FLOAT_EQ(-0.5, a.val_);
  EXPECT_FLOAT_EQ(-1.0, a.d_);
}

TEST(AgradFwdOperatorUnaryMinus, FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  fvar<var> a = -x;

  EXPECT_FLOAT_EQ(-0.5, a.val_.val());
  EXPECT_FLOAT_EQ(-1.3, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-1, g[0]);
}
TEST(AgradFwdOperatorUnaryMinus, FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  fvar<var> a = -x;

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdOperatorUnaryMinus, FvarFvarDouble) {
  using stan::agrad::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > z = -x;
  EXPECT_FLOAT_EQ(-0.5, z.val_.val_);
  EXPECT_FLOAT_EQ(-1.0, z.val_.d_);
  EXPECT_FLOAT_EQ(0, z.d_.val_);
  EXPECT_FLOAT_EQ(0, z.d_.d_);
}
TEST(AgradFwdOperatorUnaryMinus, FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > z = -x;
  EXPECT_FLOAT_EQ(-0.5, z.val_.val_.val());
  EXPECT_FLOAT_EQ(-1.0, z.val_.d_.val());
  EXPECT_FLOAT_EQ(0, z.d_.val_.val());
  EXPECT_FLOAT_EQ(0, z.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  z.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-1.0, g[0]);
}
TEST(AgradFwdOperatorUnaryMinus, FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > z = -x;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  z.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdOperatorUnaryMinus, FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > z = -x;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  z.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}


struct neg_fun {
  template <typename T0>
  inline T0
  operator()(T0 arg1) const {
    return (-arg1);
  }
};

TEST(AgradFwdOperatorUnaryMinus, neg_nan) {
  neg_fun neg_;

  test_nan(neg_,false);
}
