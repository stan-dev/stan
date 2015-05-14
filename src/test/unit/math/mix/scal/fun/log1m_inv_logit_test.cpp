#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/log1m_inv_logit.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/log1m_inv_logit.hpp>
#include <stan/math/fwd/scal/fun/log1p.hpp>
#include <stan/math/rev/scal/fun/log1p.hpp>

TEST(AgradFwdLog1mInvLogit,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1m_inv_logit;
  using std::exp;

  fvar<var> x(0.5,1.3);
  fvar<var> a = log1m_inv_logit(x);

  EXPECT_FLOAT_EQ(log1m_inv_logit(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(-1.3 * exp(0.5) / (1 + exp(0.5)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-exp(0.5) / (1 + exp(0.5)), g[0]);
}
TEST(AgradFwdLog1mInvLogit,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1m_inv_logit;
  using std::exp;

  fvar<var> x(0.5,1.3);
  fvar<var> a = log1m_inv_logit(x);

  EXPECT_FLOAT_EQ(-1.3 * exp(0.5) / (1 + exp(0.5)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.3 * exp(0.5) / (1 + exp(0.5)) / (1 + exp(0.5)), g[0]);
}
TEST(AgradFwdLog1mInvLogit,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1m_inv_logit;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = log1m_inv_logit(x);

  EXPECT_FLOAT_EQ(log1m_inv_logit(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-exp(0.5) / (1 + exp(0.5)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-exp(0.5) / (1 + exp(0.5)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = log1m_inv_logit(y);
  EXPECT_FLOAT_EQ(log1m_inv_logit(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(-exp(0.5) / (1 + exp(0.5)), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-exp(0.5) / (1 + exp(0.5)), r[0]);
}
TEST(AgradFwdLog1mInvLogit,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1m_inv_logit;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = log1m_inv_logit(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-exp(0.5) / (1 + exp(0.5)) / (1 + exp(0.5)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = log1m_inv_logit(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-exp(0.5) / (1 + exp(0.5)) / (1 + exp(0.5)), r[0]);
}
TEST(AgradFwdLog1mInvLogit,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = log1m_inv_logit(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.057556793, g[0]);
}

struct log1m_inv_logit_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log1m_inv_logit(arg1);
  }
};

TEST(AgradFwdLog1mInvLogit,log1m_inv_logit_NaN) {
  log1m_inv_logit_fun log1m_inv_logit_;
  test_nan_mix(log1m_inv_logit_,false);
}
