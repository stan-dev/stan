#include <stan/agrad/rev/log_loss.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,log_loss_zero) {
  AVAR y_hat = 0.2;
  int y = 0;
  AVAR f = stan::agrad::log_loss(y,y_hat);
  EXPECT_FLOAT_EQ(-log(1.0 - 0.2), f.val());

  AVEC x = createAVEC(y_hat);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ((1.0 / (1.0 - 0.2)), grad_f[0]);
}

TEST(AgradRev,log_loss_one) {
  AVAR y_hat = 0.2;
  int y = 1;
  AVAR f = stan::agrad::log_loss(y,y_hat);
  EXPECT_FLOAT_EQ(-log(0.2), f.val());

  AVEC x = createAVEC(y_hat);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-1.0 / 0.2, grad_f[0]);
}
