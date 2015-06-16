#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/binary_log_loss.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/binary_log_loss.hpp>

double deriv(const int y, const double y_hat) {
  if (y == 0)
    return 1.0 / (1.0 - y_hat);
  else
    return -1.0 / y_hat;
}

double deriv_2(const int y, const double y_hat) {
  if (y == 0)
    return 1.0 / ((1.0 - y_hat) * (1.0 - y_hat));
  else
    return 1.0 / (y_hat * y_hat);
}

double deriv_3(const int y, const double y_hat) {
  if (y == 0)
    return - 2.0 / pow(y_hat - 1.0, 3);
  else
    return - 2.0 / pow(y_hat, 3);
}


double finite_diff(const int y, const double y_hat) {
  using stan::math::binary_log_loss;
  static const double h = 1e-10;

  double p = binary_log_loss(y, y_hat+h);
  double m = binary_log_loss(y, y_hat-h);
  
  return (p - m) / (2 * h);
}

double finite_diff_2(const int y, const double y_hat) {
  using stan::math::binary_log_loss;
  static const double h = 1e-5;

  double p = binary_log_loss(y, y_hat+h);
  double f = binary_log_loss(y, y_hat);
  double m = binary_log_loss(y, y_hat-h);

  return exp(log(p - 2.0 * f + m) - 2.0 * log(h));
}


TEST(AgradFwdBinaryLogLoss,Fvar) {
  using stan::math::fvar;
  using stan::math::binary_log_loss;


  int y;
  fvar<double> y_hat;
  fvar<double> f;
  
  y = 0;
  y_hat = fvar<double>(0.0,1.0);
  f = binary_log_loss(y, y_hat);
  EXPECT_FLOAT_EQ(binary_log_loss(y,0.0), f.val_);
  EXPECT_FLOAT_EQ(deriv(y,0.0), f.d_);
  
  y = 1;
  y_hat = fvar<double>(1.0,1.0);
  f = binary_log_loss(y, y_hat);
  EXPECT_FLOAT_EQ(binary_log_loss(y,1.0), f.val_);
  EXPECT_FLOAT_EQ(deriv(y, 1.0), f.d_);
  
  y = 0;
  y_hat = fvar<double>(0.5,1.0);
  f = binary_log_loss(y, y_hat);
  EXPECT_FLOAT_EQ(-std::log(0.5), f.val_);
  EXPECT_FLOAT_EQ(deriv(0, 0.5), f.d_);
  EXPECT_NEAR(finite_diff(0, 0.5), f.d_, 1e-5);

  y = 1;
  y_hat = fvar<double>(0.5,1.0);
  f = binary_log_loss(y, y_hat);
  EXPECT_FLOAT_EQ(-std::log(0.5), f.val_);
  EXPECT_FLOAT_EQ(deriv(1, 0.5), f.d_);
  EXPECT_NEAR(finite_diff(1, 0.5), f.d_, 1e-5);

  y = 0;
  y_hat = fvar<double>(0.25,1.0);
  f = binary_log_loss(y, y_hat);
  EXPECT_FLOAT_EQ(-std::log(0.75), f.val_);
  EXPECT_FLOAT_EQ(deriv(0, 0.25), f.d_);
  EXPECT_NEAR(finite_diff(0, 0.25), f.d_, 1e-5);

  y = 1;
  y_hat = fvar<double>(0.75,1.0);
  f = binary_log_loss(y, y_hat);
  EXPECT_FLOAT_EQ(-std::log(0.75), f.val_);
  EXPECT_FLOAT_EQ(deriv(1, 0.75), f.d_);
  EXPECT_NEAR(finite_diff(1, 0.75), f.d_, 1e-5);
}


TEST(AgradFwdBinaryLogLoss,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::binary_log_loss;


  fvar<fvar<double> > y;
  y.val_.val_ = 0.4;
  y.d_.val_ = 1.0;
  fvar<fvar<double> > a = binary_log_loss(0.0,y);

  EXPECT_FLOAT_EQ(binary_log_loss(0.0,0.4), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(deriv(0,0.4), a.d_.val_);
  EXPECT_NEAR(finite_diff(0,0.4), a.d_.val_, 1e-5);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > x;
  x.val_.val_ = 0.4;
  x.val_.d_ = 1.0;
  fvar<fvar<double> > b = binary_log_loss(1.0,x);

  EXPECT_FLOAT_EQ(binary_log_loss(1.0,0.4), b.val_.val_);
  EXPECT_FLOAT_EQ(deriv(1,0.4), b.val_.d_);
  EXPECT_NEAR(finite_diff(1,0.4), b.val_.d_, 1e-5);
  EXPECT_FLOAT_EQ(0, b.d_.val_);
  EXPECT_FLOAT_EQ(0, b.d_.d_);
}


struct binary_log_loss_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return binary_log_loss(0,arg1);
  }
};

TEST(AgradFwdBinaryLogLoss,binary_log_loss_NaN) {
  binary_log_loss_fun binary_log_loss_;
  test_nan_fwd(binary_log_loss_,false);
}
