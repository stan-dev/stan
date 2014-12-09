#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/math/functions/binary_log_loss.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

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
  using stan::agrad::fvar;
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

TEST(AgradFwdBinaryLogLoss,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::binary_log_loss;

  fvar<var> z(0.4,3.0);
  fvar<var> a = binary_log_loss(0,z);

  EXPECT_FLOAT_EQ(binary_log_loss(0.0, 0.4), a.val_.val());
  EXPECT_FLOAT_EQ(3.0 * deriv(0, 0.4), a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(deriv(0, 0.4), g[0]);
  EXPECT_NEAR(finite_diff(0, 0.4), g[0], 1e-5);
}

TEST(AgradFwdBinaryLogLoss,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::binary_log_loss;

  fvar<var> z(0.4,3.0);
  fvar<var> a = binary_log_loss(0,z);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(3.0 * deriv_2(0,0.4), g[0]);
  EXPECT_NEAR(3.0 * finite_diff_2(0,0.4), g[0], 1e-5);
}
TEST(AgradFwdBinaryLogLoss,FvarFvarDouble) {
  using stan::agrad::fvar;
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

TEST(AgradFwdBinaryLogLoss,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::binary_log_loss;

  fvar<fvar<var> > y;
  y.val_.val_ = 0.4;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = binary_log_loss(0,y);

  EXPECT_FLOAT_EQ(binary_log_loss(0.0,0.4), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(deriv(0,0.4), a.d_.val().val());
  EXPECT_NEAR(finite_diff(0,0.4), a.d_.val().val(), 1e-5);
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(deriv(0,0.4), g[0]);

  fvar<fvar<var> > x;
  x.val_.val_ = 0.4;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > b = binary_log_loss(1,x);

  EXPECT_FLOAT_EQ(binary_log_loss(1.0,0.4), b.val_.val_.val());
  EXPECT_FLOAT_EQ(deriv(1.0,0.4), b.val_.d_.val());
  EXPECT_NEAR(finite_diff(1.0,0.4), b.val_.d_.val(), 1e-5);
  EXPECT_FLOAT_EQ(0, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(deriv(1.0, 0.4), r[0]);
}

TEST(AgradFwdBinaryLogLoss,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::binary_log_loss;

  fvar<fvar<var> > y;
  y.val_.val_ = 0.4;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = binary_log_loss(0,y);

  EXPECT_FLOAT_EQ(binary_log_loss(0.0,0.4), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(deriv(0,0.4), a.d_.val().val());
  EXPECT_NEAR(finite_diff(0,0.4), a.d_.val().val(), 1e-5);
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(deriv_2(0.0, 0.4), g[0]);

  fvar<fvar<var> > x;
  x.val_.val_ = 0.4;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > b = binary_log_loss(1,x);

  EXPECT_FLOAT_EQ(binary_log_loss(1.0,0.4), b.val_.val_.val());
  EXPECT_FLOAT_EQ(-2.5, b.val_.d_.val());
  EXPECT_FLOAT_EQ(0, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  b.val_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(deriv_2(1.0, 0.4), r[0]);
}

TEST(AgradFwdBinaryLogLoss,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::binary_log_loss;

  fvar<fvar<var> > y;
  y.val_.val_ = 0.4;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = binary_log_loss(0,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(9.2592592, g[0]);
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
  test_nan(binary_log_loss_,false);
}
