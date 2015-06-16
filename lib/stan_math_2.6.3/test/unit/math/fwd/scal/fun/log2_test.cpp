#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/log2.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/log2.hpp>

TEST(AgradFwdLog2,Fvar) {
  using stan::math::fvar;
  using std::log;
  using std::isnan;
  using stan::math::log2;

  fvar<double> x(0.5,1.0);
  
  fvar<double> a = log2(x);
  EXPECT_FLOAT_EQ(log2(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (0.5 * log(2)), a.d_);

  fvar<double> b = 2 * log2(x) + 4;
  EXPECT_FLOAT_EQ(2 * log2(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / (0.5 * log(2)), b.d_);

  fvar<double> c = -log2(x) + 5;
  EXPECT_FLOAT_EQ(-log2(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (0.5 * log(2)), c.d_);

  fvar<double> d = -3 * log2(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * log2(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (0.5 * log(2)) + 5, d.d_);

  fvar<double> y(-0.5,1.0);
  fvar<double> e = log2(y);
  isnan(e.val_);
  isnan(e.d_);

  fvar<double> z(0.0,1.0);
  fvar<double> f = log2(z);
  isnan(f.val_);
  isnan(f.d_);
}


TEST(AgradFwdLog2,FvarFvarDouble) {
  using stan::math::fvar;
  using std::log;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = log2(x);

  EXPECT_FLOAT_EQ(log2(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(1 / (0.5 * log(2)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = log2(y);
  EXPECT_FLOAT_EQ(log2(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(1 / (0.5 * log(2)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct log2_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log2(arg1);
  }
};

TEST(AgradFwdLog2,log2_NaN) {
  log2_fun log2_;
  test_nan_fwd(log2_,false);
}
