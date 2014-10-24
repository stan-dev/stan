#include <gtest/gtest.h>
#include <stan/math/functions/trigamma.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdTrigamma, Fvar) {
  using stan::agrad::fvar;
  using stan::math::trigamma;

  fvar<double> x(0.5,1.0);
  fvar<double> a = trigamma(x);
  EXPECT_FLOAT_EQ(4.9348022005446793094, a.val_);
  EXPECT_FLOAT_EQ(-16.8288, a.d_);
}
TEST(AgradFwdTrigamma, FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::trigamma;  
  
  fvar<var> x(0.5,1.3);
  fvar<var> a = trigamma(x);

  EXPECT_FLOAT_EQ(4.9348022005446793094, a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * -16.8288, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-16.8288, g[0]);
}
TEST(AgradFwdTrigamma, FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::trigamma;  
  
  fvar<var> x(0.5,1.3);
  fvar<var> a = trigamma(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(126.63182, g[0]);
}
TEST(AgradFwdTrigamma, FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::trigamma;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = trigamma(x);

  EXPECT_FLOAT_EQ(4.9348022005446793094, a.val_.val_);
  EXPECT_FLOAT_EQ(-16.8288, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = trigamma(y);
  EXPECT_FLOAT_EQ(4.9348022005446793094, a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(-16.8288, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdTrigamma, FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::trigamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = trigamma(x);

  EXPECT_FLOAT_EQ(4.9348022005446793094, a.val_.val_.val());
  EXPECT_FLOAT_EQ(-16.8288, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-16.8288, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = trigamma(y);
  EXPECT_FLOAT_EQ(4.9348022005446793094, b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(-16.8288, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-16.8288, r[0]);
}
TEST(AgradFwdTrigamma, FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::trigamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = trigamma(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(97.409088, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = trigamma(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(97.409088, r[0]);
}
TEST(AgradFwdTrigamma, FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::trigamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = trigamma(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-771.47424982666722519053592192, g[0]);
}

struct trigamma_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return stan::math::trigamma(arg1);
  }
};

TEST(AgradFwdTrigamma,trigamma_NaN) {
  trigamma_fun trigamma_;
  test_nan(trigamma_,false);
}
