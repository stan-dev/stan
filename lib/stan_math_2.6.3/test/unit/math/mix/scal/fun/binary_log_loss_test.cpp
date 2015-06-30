#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/binary_log_loss.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/binary_log_loss.hpp>
#include <stan/math/rev/scal/fun/binary_log_loss.hpp>

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




TEST(AgradFwdBinaryLogLoss,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binary_log_loss;

  fvar<var> z(0.4,3.0);
  fvar<var> a = binary_log_loss(0,z);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(3.0 * deriv_2(0,0.4), g[0]);
  EXPECT_NEAR(3.0 * finite_diff_2(0,0.4), g[0], 1e-5);
}


TEST(AgradFwdBinaryLogLoss,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  test_nan_mix(binary_log_loss_,false);
}
