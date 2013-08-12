#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/log_sum_exp.hpp>
#include <stan/agrad/fwd/log_sum_exp.hpp>

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

void log_sum_exp_test(const std::vector<double>& x) {
  using std::exp;
  using stan::agrad::fvar;
  for (size_t n = 0; n < x.size(); ++n) {
    // for d/d.x[n]
    std::vector<fvar<double> > xv(x.size());
    for (size_t i = 0; i < x.size(); ++i)
      xv[i] = x[i];
    xv[n].d_ = 2.3;
    fvar<double> sum_exp = 0;
    for (size_t i = 0; i < x.size(); ++i)
      sum_exp += exp(xv[i]);
    fvar<double> log_sum_exp_expected = log(sum_exp);
    double val_expected = log_sum_exp_expected.val_;
    double deriv_expected = log_sum_exp_expected.d_;
  
    std::vector<fvar<double> > xv2(x.size());
    for (size_t i = 0; i < x.size(); ++i)
      xv2[i] = x[i];
    xv2[n].d_ = 2.3;
    fvar<double> log_sum_exp_fvar = log_sum_exp(xv2);
    double val = log_sum_exp_fvar.val_;
    double deriv = log_sum_exp_fvar.d_;
    
    EXPECT_FLOAT_EQ(val_expected, val);
    EXPECT_FLOAT_EQ(deriv_expected, deriv);
  }
}

TEST(AgradRevLogSumExp,vector) {
  using std::vector;

  vector<double> a(1);
  a[0] = 1.3;
  log_sum_exp_test(a);

  vector<double> b(3);
  b[0] = -1.2;
  b[1] = 0;
  b[2] = 3.9;
  log_sum_exp_test(b);
}
