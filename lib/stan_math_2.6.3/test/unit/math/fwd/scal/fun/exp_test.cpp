#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>

TEST(AgradFwdExp,Fvar) {
  using stan::math::fvar;
  using std::exp;

  fvar<double> x(0.5,1.0);
  
  fvar<double> a = exp(x);
  EXPECT_FLOAT_EQ(exp(0.5), a.val_);
  EXPECT_FLOAT_EQ(exp(0.5), a.d_);

  fvar<double> b = 2 * exp(x) + 4;
  EXPECT_FLOAT_EQ(2 * exp(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 * exp(0.5), b.d_);

  fvar<double> c = -exp(x) + 5;
  EXPECT_FLOAT_EQ(-exp(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-exp(0.5), c.d_);

  fvar<double> d = -3 * exp(-x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * exp(-0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(3 * exp(-0.5) + 5, d.d_);

  fvar<double> y(-0.5,1.0);
  fvar<double> e = exp(y);
  EXPECT_FLOAT_EQ(exp(-0.5), e.val_);
  EXPECT_FLOAT_EQ(exp(-0.5), e.d_);

  fvar<double> z(0.0,1.0);
  fvar<double> f = exp(z);
  EXPECT_FLOAT_EQ(exp(0.0), f.val_);
  EXPECT_FLOAT_EQ(exp(0.0), f.d_);
}


TEST(AgradFwdExp,FvarFvarDouble) {
  using stan::math::fvar;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = exp(x);

  EXPECT_FLOAT_EQ(exp(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = exp(y);
  EXPECT_FLOAT_EQ(exp(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(exp(0.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct exp_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return exp(arg1);
  }
};

TEST(AgradFwdExp,exp_NaN) {
  exp_fun exp_;
  test_nan_fwd(exp_,false);
}
