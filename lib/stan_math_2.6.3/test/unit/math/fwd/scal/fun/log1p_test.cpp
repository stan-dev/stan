#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/log1p.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/log1p.hpp>

TEST(AgradFwdLog1p,Fvar) {
  using stan::math::fvar;
  using stan::math::log1p;
  using std::isnan;

  fvar<double> x(0.5,1.0);
  fvar<double> y(-1.0,2.0);
  fvar<double> z(-2.0,3.0);

  fvar<double> a = log1p(x);
  EXPECT_FLOAT_EQ(log1p(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 + 0.5), a.d_);

  fvar<double> b = log1p(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> c = log1p(z);
  isnan(c.val_);
  isnan(c.d_);
}

TEST(AgradFwdLog1p,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::log1p;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = log1p(x);

  EXPECT_FLOAT_EQ(log1p(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(1 / (1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = log1p(y);
  EXPECT_FLOAT_EQ(log1p(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(1 / (1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct log1p_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log1p(arg1);
  }
};

TEST(AgradFwdLog1p,log1p_NaN) {
  log1p_fun log1p_;
  test_nan_fwd(log1p_,false);
}
