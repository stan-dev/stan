#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/log_diff_exp.hpp>

TEST(AgradFvar, log_diff_exp) {
  using stan::agrad::fvar;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<double> x(1.2);
  fvar<double> y(0.5);
  x.d_ = 1.0;
  y.d_ = 2.0;

  double z = 1.1;

  fvar<double> a = log_diff_exp(x, y);
  EXPECT_FLOAT_EQ(log_diff_exp(1.2, 0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 - exp(0.5 - 1.2) ) + 2 / (1 - exp(1.2 - 0.5) ), a.d_);

  fvar<double> b = log_diff_exp(x, z);
  EXPECT_FLOAT_EQ(log_diff_exp(1.2, 1.1), b.val_);
  EXPECT_FLOAT_EQ(1 / (1 - exp(1.1 - 1.2) ), b.d_);

  fvar<double> c = log_diff_exp(z, y);
  EXPECT_FLOAT_EQ(log_diff_exp(1.1, 0.5), c.val_);
  EXPECT_FLOAT_EQ(2 / (1 - exp(1.1 - 0.5) ), c.d_);
}

TEST(AgradFvar,log_diff_exp_exception) {
  using stan::agrad::fvar;
  EXPECT_NO_THROW(log_diff_exp(fvar<double>(3), fvar<double>(4)));
  EXPECT_NO_THROW(log_diff_exp(fvar<double>(3), 4));
  EXPECT_NO_THROW(log_diff_exp(3, fvar<double>(4)));
}