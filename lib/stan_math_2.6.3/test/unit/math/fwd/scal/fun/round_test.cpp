#include <gtest/gtest.h>
#include <boost/math/special_functions/round.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/round.hpp>

TEST(AgradFwdRound, Fvar) {
  using stan::math::fvar;
  using boost::math::round;

  fvar<double> x(0.5,1.0);
  fvar<double> y(2.4,2.0);

  fvar<double> a = round(x);
  EXPECT_FLOAT_EQ(round(0.5), a.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_);

  fvar<double> b = round(y);
  EXPECT_FLOAT_EQ(round(2.4), b.val_);
  EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c = round(2 * x);
  EXPECT_FLOAT_EQ(round(2 * 0.5), c.val_);
  EXPECT_FLOAT_EQ(0.0, c.d_);

  fvar<double> z(1.25,1.0);

  fvar<double> d = round(2 * z);
  EXPECT_FLOAT_EQ(round(2 * 1.25), d.val_);
   EXPECT_FLOAT_EQ(0.0, d.d_);
}

TEST(AgradFwdRound, FvarFvarDouble) {
  using stan::math::fvar;
  using boost::math::round;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = round(x);

  EXPECT_FLOAT_EQ(round(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = round(y);
  EXPECT_FLOAT_EQ(round(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct round_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return round(arg1);
  }
};

TEST(AgradFwdRound,round_NaN) {
  round_fun round_;
  test_nan_fwd(round_,false);
}
