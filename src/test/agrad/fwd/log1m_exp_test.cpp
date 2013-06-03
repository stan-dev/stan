#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/log1m_exp.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, log1m_exp) {
  using stan::agrad::fvar;
  using stan::math::log1m_exp;
  using std::exp;
  using std::log;

  fvar<double> x(-0.5);
  fvar<double> y(-1.0);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = log1m_exp(x);
  EXPECT_FLOAT_EQ(log1m_exp(-0.5), a.val_);
  EXPECT_FLOAT_EQ(-exp(-0.5) / (1 - exp(-0.5)), a.d_);
  EXPECT_FLOAT_EQ(-1 / boost::math::expm1(0.5), a.d_);

  fvar<double> b = log1m_exp(y);
  EXPECT_FLOAT_EQ(log1m_exp(-1.0), b.val_);
  EXPECT_FLOAT_EQ(2.0 * -exp(-1.0) / (1 - exp(-1.0)), b.d_);
  EXPECT_FLOAT_EQ(2.0 * -1 / boost::math::expm1(1), b.d_);
  
  fvar<double> a2 = log(1-exp(x));
  EXPECT_FLOAT_EQ(a.d_, a2.d_);

  fvar<double> b2 = log(1-exp(y));
  EXPECT_FLOAT_EQ(b.d_, b2.d_);
}

TEST(AgradFvar, log1m_exp_exception) {
  using stan::agrad::fvar;
  using stan::math::log1m_exp;
  EXPECT_NO_THROW(log1m_exp(fvar<double>(-3)));
  EXPECT_NO_THROW(log1m_exp(fvar<double>(3)));
}

TEST(AgradFvarVar, log1m_exp) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log1m_exp;
  using std::exp;

  fvar<var> x(-0.2,1.3);
  fvar<var> a = log1m_exp(x);

  EXPECT_FLOAT_EQ(log1m_exp(-0.2), a.val_.val());
  EXPECT_FLOAT_EQ(-1.3 * exp(-0.2) / (1.0 - exp(-0.2)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-exp(-0.2) / (1.0 - exp(-0.2)),g[0]);
}

TEST(AgradFvarFvar, log1m_exp) {
  using stan::agrad::fvar;
  using stan::math::log1m_exp;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = -0.2;
  x.val_.d_ = 1.0;
  fvar<fvar<double> > a = log1m_exp(x);

  EXPECT_FLOAT_EQ(log1m_exp(-0.2), a.val_.val_);
  EXPECT_FLOAT_EQ(-exp(-0.2) / (1.0 - exp(-0.2)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
