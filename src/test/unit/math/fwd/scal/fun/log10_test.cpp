#include <gtest/gtest.h>
#include <stan/math/fwd/scal/fun/log10.hpp>
#include <stan/math/fwd/core.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>

TEST(AgradFwdLog10,Fvar) {
  using stan::math::fvar;
  using std::log;
  using std::isnan;
  using std::log10;

  fvar<double> x(0.5,1.0);
  
  fvar<double> a = log10(x);
  EXPECT_FLOAT_EQ(log10(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (0.5 * log(10)), a.d_);

  fvar<double> b = 2 * log10(x) + 4;
  EXPECT_FLOAT_EQ(2 * log10(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / (0.5 * log(10)), b.d_);

  fvar<double> c = -log10(x) + 5;
  EXPECT_FLOAT_EQ(-log10(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (0.5 * log(10)), c.d_);

  fvar<double> d = -3 * log10(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * log10(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (0.5 * log(10)) + 5, d.d_);

  fvar<double> y(-0.5,1.0);
  fvar<double> e = log10(y);
  isnan(e.val_);
  isnan(e.d_);

  fvar<double> z(0.0,1.0);
  fvar<double> f = log10(z);
  isnan(f.val_);
  isnan(f.d_);
}

TEST(AgradFwdLog10,FvarFvarDouble) {
  using stan::math::fvar;
  using std::log;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = log10(x);

  EXPECT_FLOAT_EQ(log10(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(1 / (0.5 * log(10)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = log10(y);
  EXPECT_FLOAT_EQ(log10(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(1 / (0.5 * log(10)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct log10_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log10(arg1);
  }
};

TEST(AgradFwdLog10,log10_NaN) {
  log10_fun log10_;
  test_nan_fwd(log10_,false);
}
