#include <gtest/gtest.h>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/lgamma.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>

TEST(AgradFwdLgamma,Fvar) {
  using stan::math::fvar;
  using boost::math::lgamma;
  using boost::math::digamma;

  fvar<double> x(0.5,1.0);

  fvar<double> a = lgamma(x);
  EXPECT_FLOAT_EQ(lgamma(0.5), a.val_);
  EXPECT_FLOAT_EQ(digamma(0.5), a.d_);
}

TEST(AgradFwdLgamma,FvarFvarDouble) {
  using stan::math::fvar;
  using boost::math::lgamma;
  using boost::math::digamma;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = lgamma(x);

  EXPECT_FLOAT_EQ(lgamma(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(digamma(0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = lgamma(y);
  EXPECT_FLOAT_EQ(lgamma(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(digamma(0.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct lgamma_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return lgamma(arg1);
  }
};

TEST(AgradFwdLgamma,lgamma_NaN) {
  lgamma_fun lgamma_;
  test_nan_fwd(lgamma_,false);
}
