#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/math/functions/lmgamma.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdLmgamma,Fvar) {
  using stan::agrad::fvar;
  using stan::math::lmgamma;

  int x = 3;
  fvar<double> y(3.2,2.1);

  fvar<double> a = lmgamma(x, y);
  EXPECT_FLOAT_EQ(lmgamma(3, 3.2), a.val_);
  EXPECT_FLOAT_EQ(4.9138227, a.d_);
}

TEST(AgradFwdLmgamma,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::lmgamma;

  fvar<var> x(3.2,2.1);
  fvar<var> a = lmgamma(3, x);

  EXPECT_FLOAT_EQ(lmgamma(3,3.2), a.val_.val());
  EXPECT_FLOAT_EQ(4.9138227, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(4.9138227 / 2.1, g[0]);
}
TEST(AgradFwdLmgamma,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::lmgamma;

  fvar<var> x(3.2,2.1);
  fvar<var> a = lmgamma(3, x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(2.9115787, g[0]);
}
TEST(AgradFwdLmgamma,FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::lmgamma;

  fvar<fvar<double> > x;
  x.val_.val_ = 3.2;
  x.val_.d_ = 2.1;

  fvar<fvar<double> > a = lmgamma(3,x);

  EXPECT_FLOAT_EQ(lmgamma(3,3.2), a.val_.val_);
  EXPECT_FLOAT_EQ(4.9138227, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 3.2;
  y.d_.val_ = 2.1;

  a = lmgamma(3,y);
  EXPECT_FLOAT_EQ(lmgamma(3,3.2), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(4.9138227, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdLmgamma,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::lmgamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.2;
  x.val_.d_ = 2.1;

  fvar<fvar<var> > a = lmgamma(3,x);

  EXPECT_FLOAT_EQ(lmgamma(3,3.2), a.val_.val_.val());
  EXPECT_FLOAT_EQ(4.9138227, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(4.9138227 / 2.1, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 3.2;
  y.d_.val_ = 2.1;

  fvar<fvar<var> > b = lmgamma(3,y);
  EXPECT_FLOAT_EQ(lmgamma(3,3.2), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(4.9138227, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(4.9138227 / 2.1, r[0]);
}
TEST(AgradFwdLmgamma,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::lmgamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.2;
  x.val_.d_ = 2.1;

  fvar<fvar<var> > a = lmgamma(3,x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(2.9115787, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 3.2;
  y.d_.val_ = 2.1;

  fvar<fvar<var> > b = lmgamma(3,y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.9115787, r[0]);
}
TEST(AgradFwdLmgamma,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.2;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = stan::math::lmgamma(3,x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.65043455, g[0]);
}

struct lmgamma_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return lmgamma(3,arg1);
  }
};

TEST(AgradFwdLmgamma,lmgamma_NaN) {
  lmgamma_fun lmgamma_;
  test_nan(lmgamma_,false);
}
