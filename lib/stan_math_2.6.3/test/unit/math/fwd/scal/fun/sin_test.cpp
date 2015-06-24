#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>

TEST(AgradFwdSin, Fvar) {
  using stan::math::fvar;
  using std::sin;
  using std::cos;

  fvar<double> x(0.5,1.0);
  
  fvar<double> a = sin(x);
  EXPECT_FLOAT_EQ(sin(0.5), a.val_);
  EXPECT_FLOAT_EQ(cos(0.5), a.d_);

  fvar<double> b = 2 * sin(x) + 4;
  EXPECT_FLOAT_EQ(2 * sin(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 * cos(0.5), b.d_);

  fvar<double> c = -sin(x) + 5;
  EXPECT_FLOAT_EQ(-sin(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-cos(0.5), c.d_);

  fvar<double> d = -3 * sin(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * sin(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 * cos(0.5) + 5, d.d_);

  fvar<double> y(-0.5,1.0);
  fvar<double> e = sin(y);
  EXPECT_FLOAT_EQ(sin(-0.5), e.val_);
  EXPECT_FLOAT_EQ(cos(-0.5), e.d_);

  fvar<double> z(0.0,1.0);
  fvar<double> f = sin(z);
  EXPECT_FLOAT_EQ(sin(0.0), f.val_);
  EXPECT_FLOAT_EQ(cos(0.0), f.d_);
}

TEST(AgradFwdSin, FvarFvarDouble) {
  using stan::math::fvar;
  using std::sin;
  using std::cos;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = sin(x);

  EXPECT_FLOAT_EQ(sin(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2.0 * cos(1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = sin(y);
  EXPECT_FLOAT_EQ(sin(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2.0 * cos(1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct sin_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return sin(arg1);
  }
};

TEST(AgradFwdSin,sin_NaN) {
  sin_fun sin_;
  test_nan_fwd(sin_,false);
}
