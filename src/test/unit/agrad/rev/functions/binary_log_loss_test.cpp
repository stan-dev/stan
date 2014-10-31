#include <stan/agrad/rev/functions/binary_log_loss.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/functions/binary_log_loss.hpp>
#include <test/unit/agrad/rev/nan_util.hpp>

double inf = std::numeric_limits<double>::infinity();

double deriv(const int y, const double y_hat) {
  if (y == 0)
    return 1.0 / (1.0 - y_hat);
  else
    return -1.0 / y_hat;
}

double finite_diff(const int y, const double y_hat) {
  using stan::math::binary_log_loss;
  static const double e = 1e-10;

  double p = binary_log_loss(y, y_hat+e);
  double m = binary_log_loss(y, y_hat-e);
  
  return (p - m) / (2 * e);
}

TEST(AgradRev,binary_log_loss) {
  using std::log;

  int y;
  AVAR y_hat, f;
  AVEC x;
  VEC grad_f;
  
  y = 0;
  y_hat = 0.0;
  x = createAVEC(y_hat);
  f = stan::agrad::binary_log_loss(y, y_hat);
  f.grad(x, grad_f);
  EXPECT_FLOAT_EQ(0.0, f.val());
  EXPECT_FLOAT_EQ(deriv(0, 0.0), grad_f[0]);

  y = 1;
  y_hat = 1.0;
  x = createAVEC(y_hat);
  f = stan::agrad::binary_log_loss(y, y_hat);
  f.grad(x, grad_f);
  EXPECT_FLOAT_EQ(0.0, f.val());
  EXPECT_FLOAT_EQ(deriv(1, 1.0), grad_f[0]);

  y = 0;
  y_hat = 0.5;
  x = createAVEC(y_hat);
  f = stan::agrad::binary_log_loss(y, y_hat);
  f.grad(x, grad_f);
  EXPECT_FLOAT_EQ(-std::log(0.5), f.val());
  EXPECT_FLOAT_EQ(deriv(0, 0.5), grad_f[0]);
  EXPECT_NEAR(finite_diff(0, 0.5), grad_f[0], 1e-5);

  y = 1;
  y_hat = 0.5;
  x = createAVEC(y_hat);
  f = stan::agrad::binary_log_loss(y, y_hat);
  f.grad(x, grad_f);
  EXPECT_FLOAT_EQ(-std::log(0.5), f.val());
  EXPECT_FLOAT_EQ(deriv(1, 0.5), grad_f[0]);
  EXPECT_NEAR(finite_diff(1, 0.5), grad_f[0], 1e-5);

  y = 0;
  y_hat = 0.25;
  x = createAVEC(y_hat);
  f = stan::agrad::binary_log_loss(y, y_hat);
  f.grad(x, grad_f);
  EXPECT_FLOAT_EQ(-std::log(0.75), f.val());
  EXPECT_FLOAT_EQ(deriv(0, 0.25), grad_f[0]);
  EXPECT_NEAR(finite_diff(0, 0.25), grad_f[0], 1e-5);

  y = 1;
  y_hat = 0.75;
  x = createAVEC(y_hat);
  f = stan::agrad::binary_log_loss(y, y_hat);
  f.grad(x, grad_f);
  EXPECT_FLOAT_EQ(-std::log(0.75), f.val());
  EXPECT_FLOAT_EQ(deriv(1, 0.75), grad_f[0]);
  EXPECT_NEAR(finite_diff(1, 0.75), grad_f[0], 1e-5);
}

struct binary_log_loss_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return binary_log_loss(1,arg1);
  }
};

TEST(AgradRev,binary_log_loss_NaN) {
  binary_log_loss_fun binary_log_loss_;
  test_nan(binary_log_loss_,false,true);
}
