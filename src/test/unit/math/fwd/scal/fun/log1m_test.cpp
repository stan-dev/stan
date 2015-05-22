#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/log1m.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>

TEST(AgradFwdLog1m,Fvar) {
  using stan::math::fvar;
  using stan::math::log1m;
  using std::isnan;

  fvar<double> x(0.5,1.0);
  fvar<double> y(1.0,2.0);
  fvar<double> z(2.0,3.0);

  fvar<double> a = log1m(x);
  EXPECT_FLOAT_EQ(log1m(0.5), a.val_);
  EXPECT_FLOAT_EQ(-1 / (1 - 0.5), a.d_);

  fvar<double> b = log1m(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> c = log1m(z);
  isnan(c.val_);
  isnan(c.d_);
}

TEST(AgradFwdLog1m,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::log1m;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = log1m(x);

  EXPECT_FLOAT_EQ(log1m(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(-1 / (0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = log1m(y);
  EXPECT_FLOAT_EQ(log1m(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(-1 / (0.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct log1m_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log1m(arg1);
  }
};

TEST(AgradFwdLog1m,log1m_NaN) {
  log1m_fun log1m_;
  test_nan_fwd(log1m_,false);
}
