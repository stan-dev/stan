#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/log1m_exp.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/log1m_exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/expm1.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log1m_exp.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>


TEST(AgradFwdLog1mExp,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1m_exp;
  using std::exp;

  fvar<var> x(-0.2,1.3);
  fvar<var> a = log1m_exp(x);

  EXPECT_FLOAT_EQ(log1m_exp(-0.2), a.val_.val());
  EXPECT_FLOAT_EQ(-1.3 * exp(-0.2) / (1.0 - exp(-0.2)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-exp(-0.2) / (1.0 - exp(-0.2)),g[0]);
}
TEST(AgradFwdLog1mExp,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1m_exp;
  using std::exp;

  fvar<var> x(-0.2,1.3);
  fvar<var> a = log1m_exp(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * -exp(-0.2) / (1.0 - exp(-0.2)) / (1.0 - exp(-0.2)),g[0]);
}
TEST(AgradFwdLog1mExp,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1m_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = -0.2;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > a = log1m_exp(x);

  EXPECT_FLOAT_EQ(log1m_exp(-0.2), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-exp(-0.2) / (1.0 - exp(-0.2)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-exp(-0.2) / (1.0 - exp(-0.2)), g[0]);
}
TEST(AgradFwdLog1mExp,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log1m_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = -0.2;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > a = log1m_exp(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-exp(-0.2) / (1.0 - exp(-0.2)) / (1.0 - exp(-0.2)),g[0]);
}
TEST(AgradFwdLog1mExp,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = -0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = log1m_exp(x);
  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-15.995914931852, g[0]);
}

struct log1m_exp_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log1m_exp(arg1);
  }
};

TEST(AgradFwdLog1mExp,log1m_exp_NaN) {
  log1m_exp_fun log1m_exp_;
  test_nan_mix(log1m_exp_,false);
}
