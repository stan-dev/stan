#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/log_diff_exp.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

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

TEST(AgradFvarVar, log_diff_exp) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<var> x(9.0,1.3);
  fvar<var> z(6.0,1.0);
  fvar<var> a = log_diff_exp(x,z);

  EXPECT_FLOAT_EQ(log_diff_exp(9.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.3 * exp(9.0) - 1.0 * exp(6.0)) / (exp(9.0) - exp(6.0)), a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(9.0) / (exp(9.0) - exp(6.0)),g[0]);
  EXPECT_FLOAT_EQ(-exp(6.0) / (exp(9.0) - exp(6.0)),g[1]);
}

TEST(AgradFvarFvar, log_diff_exp) {
  using stan::agrad::fvar;
  using stan::math::log_diff_exp;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = 9.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = log_diff_exp(x,y);

  EXPECT_FLOAT_EQ(log_diff_exp(9.0,6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(9.0) / (exp(9.0) - exp(6.0)), a.val_.d_);
  EXPECT_FLOAT_EQ(-exp(6.0) / (exp(9.0) - exp(6.0)), a.d_.val_);
  EXPECT_FLOAT_EQ(0.055141006, a.d_.d_);
}
