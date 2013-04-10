#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/log1m_inv_logit.hpp>

TEST(AgradFvar, log1m_inv_logit){
  using stan::agrad::fvar;
  using stan::math::log1m_inv_logit;
  using std::exp;

  fvar<double> x(0.5);
  fvar<double> y(-1.0);
  fvar<double> z(0.0);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

  fvar<double> a = log1m_inv_logit(x);
  EXPECT_FLOAT_EQ(log1m_inv_logit(0.5), a.val_);
  EXPECT_FLOAT_EQ(-1.0 * exp(0.5) / (1 + exp(0.5)), a.d_);

  fvar<double> b = log1m_inv_logit(y);
  EXPECT_FLOAT_EQ(log1m_inv_logit(-1.0), b.val_);
  EXPECT_FLOAT_EQ(-2.0 * exp(-1.0) / (1 + exp(-1.0)), b.d_);

  fvar<double> c = log1m_inv_logit(z);
  EXPECT_FLOAT_EQ(log1m_inv_logit(0.0), c.val_);
  EXPECT_FLOAT_EQ(-3.0 * exp(0.0) / (1 + exp(0.0)), c.d_);
}
