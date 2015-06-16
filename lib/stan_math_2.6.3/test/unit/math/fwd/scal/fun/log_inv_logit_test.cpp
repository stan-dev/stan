#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/log_inv_logit.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/log_inv_logit.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/log1p.hpp>

TEST(AgradFwdLogInvLogit,Fvar) {
  using stan::math::fvar;
  using stan::math::log_inv_logit;
  using std::exp;

  fvar<double> x(0.5,1.0);
  fvar<double> y(-1.0,2.0);
  fvar<double> z(0.0,3.0);

  fvar<double> a = log_inv_logit(x);
  EXPECT_FLOAT_EQ(log_inv_logit(0.5), a.val_);
  EXPECT_FLOAT_EQ(1.0 * exp(-0.5) / (1 + exp(-0.5)), a.d_);

  fvar<double> b = log_inv_logit(y);
  EXPECT_FLOAT_EQ(log_inv_logit(-1.0), b.val_);
  EXPECT_FLOAT_EQ(2.0 * exp(1.0) / (1 + exp(1.0)), b.d_);

  fvar<double> c = log_inv_logit(z);
  EXPECT_FLOAT_EQ(log_inv_logit(0.0), c.val_);
  EXPECT_FLOAT_EQ(3.0 * exp(0.0) / (1 + exp(0.0)), c.d_);
}

TEST(AgradFwdLogInvLogit,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::log_inv_logit;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = log_inv_logit(x);

  EXPECT_FLOAT_EQ(log_inv_logit(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(-0.5) / (1 + exp(-0.5)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = log_inv_logit(y);
  EXPECT_FLOAT_EQ(log_inv_logit(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(exp(-0.5) / (1 + exp(-0.5)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct log_inv_logit_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log_inv_logit(arg1);
  }
};

TEST(AgradFwdLogInvLogit,log_inv_logit_NaN) {
  log_inv_logit_fun log_inv_logit_;
  test_nan_fwd(log_inv_logit_,false);
}
