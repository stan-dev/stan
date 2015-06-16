#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/log1m_exp.hpp>
#include <stan/math/fwd/scal/fun/log1m_exp.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <cmath>

TEST(AgradFwdLog1mExp,Fvar) {
  using stan::math::fvar;
  using stan::math::log1m_exp;
  using std::exp;
  using std::log;

  fvar<double> x(-0.5);
  fvar<double> y(-1.0);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = log1m_exp(x);
  EXPECT_FLOAT_EQ(log1m_exp(-0.5), a.val_);
  EXPECT_FLOAT_EQ(-exp(-0.5) / (1 - exp(-0.5)), a.d_);
  EXPECT_FLOAT_EQ(-1 / ::expm1(0.5), a.d_);

  fvar<double> b = log1m_exp(y);
  EXPECT_FLOAT_EQ(log1m_exp(-1.0), b.val_);
  EXPECT_FLOAT_EQ(2.0 * -exp(-1.0) / (1 - exp(-1.0)), b.d_);
  EXPECT_FLOAT_EQ(2.0 * -1 / ::expm1(1), b.d_);
  
  fvar<double> a2 = log(1-exp(x));
  EXPECT_FLOAT_EQ(a.d_, a2.d_);

  fvar<double> b2 = log(1-exp(y));
  EXPECT_FLOAT_EQ(b.d_, b2.d_);
}

TEST(AgradFwdLog1mExp,Fvar_exception) {
  using stan::math::fvar;
  using stan::math::log1m_exp;
  EXPECT_NO_THROW(log1m_exp(fvar<double>(-3)));
  EXPECT_NO_THROW(log1m_exp(fvar<double>(3)));
}


TEST(AgradFwdLog1mExp,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::log1m_exp;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = -0.2;
  x.val_.d_ = 1.0;
  fvar<fvar<double> > a = log1m_exp(x);

  EXPECT_FLOAT_EQ(log1m_exp(-0.2), a.val_.val_);
  EXPECT_FLOAT_EQ(-exp(-0.2) / (1.0 - exp(-0.2)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
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
  test_nan_fwd(log1m_exp_,false);
}
