#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/digamma.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>

TEST(AgradFwdDigamma,Fvar) {
  using stan::math::fvar;
  using boost::math::digamma;
  using boost::math::zeta;

  fvar<double> x(0.5,1.0);
  fvar<double> a = digamma(x);
  EXPECT_FLOAT_EQ(digamma(0.5), a.val_);
  EXPECT_FLOAT_EQ(4.9348022005446793094, a.d_);
}

TEST(AgradFwdDigamma,FvarFvarDouble) {
  using stan::math::fvar;
  using boost::math::digamma;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = digamma(x);

  EXPECT_FLOAT_EQ(digamma(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(4.9348022005446793094, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = digamma(y);
  EXPECT_FLOAT_EQ(digamma(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(4.9348022005446793094, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct digamma_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return digamma(arg1);
  }
};

TEST(AgradFwdDigamma,digamma_NaN) {
  digamma_fun digamma_;
  test_nan_fwd(digamma_,false);
}
