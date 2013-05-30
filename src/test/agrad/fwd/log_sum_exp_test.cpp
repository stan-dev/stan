#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/log_sum_exp.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, log_sum_exp) {
  using stan::agrad::fvar;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<double> x(0.5,1.0);
  fvar<double> y(1.2,2.0);
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

TEST(AgradFvarVar, log_sum_exp) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<var> x(3.0,1.3);
  fvar<var> z(6.0,1.0);
  fvar<var> a = log_sum_exp(x,z);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.3 * exp(3.0) + 1.0 * exp(6.0)) / (exp(3.0) + exp(6.0)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)),g[0]);
  std::isnan(g[1]);

  y = createAVEC(x.d_);
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0,g[0]);
  std::isnan(g[1]);
}

TEST(AgradFvarFvar, log_sum_exp) {
  using stan::agrad::fvar;
  using stan::math::log_sum_exp;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = log_sum_exp(x,y);

  EXPECT_FLOAT_EQ(log_sum_exp(3.0,6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(3.0) / (exp(3.0) + exp(6.0)), a.val_.d_);
  EXPECT_FLOAT_EQ(exp(6.0) / (exp(3.0) + exp(6.0)), a.d_.val_);
  EXPECT_FLOAT_EQ(-0.045176659, a.d_.d_);
}
