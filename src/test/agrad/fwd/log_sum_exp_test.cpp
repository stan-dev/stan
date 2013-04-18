#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/log_sum_exp.hpp>

TEST(AgradFvar, log_sum_exp) {
  using stan::agrad::fvar;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<double> x(0.5);
  fvar<double> y(1.2);
  x.d_ = 1.0;
  y.d_ = 2.0;

  double z = 1.4;

  fvar<double> a = log_sum_exp(x, y);
  EXPECT_FLOAT_EQ(log_sum_exp(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ((1.0 * exp(0.5) + 2.0 * exp(1.2)) / (exp(0.5) 
                                                       + exp(1.2)), a.d_);

  fvar<double> b = log_sum_exp(x, z);
  EXPECT_FLOAT_EQ(log_sum_exp(0.5, 1.4), b.val_);
  EXPECT_FLOAT_EQ(1.0 * exp(0.5) / (exp(0.5) + exp(1.4)), b.d_);

  fvar<double> c = log_sum_exp(z, x);
  EXPECT_FLOAT_EQ(log_sum_exp(1.4, 0.5), c.val_);
  EXPECT_FLOAT_EQ(1.0 * exp(0.5) / (exp(0.5) + exp(1.4)), c.d_);
}
