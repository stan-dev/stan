#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/inv_sqrt.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/inv_sqrt.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>

TEST(AgradFwdInvSqrt,Fvar) {
  using stan::math::fvar;
  using stan::math::inv_sqrt;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // Derivatives w.r.t. x
  fvar<double> a = inv_sqrt(x);

  EXPECT_FLOAT_EQ(inv_sqrt(0.5), a.val_);
  EXPECT_FLOAT_EQ(-0.5 / (0.5 * std::sqrt(0.5)), a.d_);

  fvar<double> y(0.0);
  y.d_ = 1.0;
  fvar<double> g = inv_sqrt(y);

  EXPECT_FLOAT_EQ(stan::math::positive_infinity(), g.val_);
  EXPECT_FLOAT_EQ(stan::math::negative_infinity(), g.d_);

  fvar<double> z(-1.0);
  z.d_ = 2.0;
  g = inv_sqrt(z);

  std::isnan(g.val_);
  std::isnan(g.d_);
}   

TEST(AgradFwdInvSqrt,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::inv_sqrt;
  using std::log;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = inv_sqrt(x);

  EXPECT_FLOAT_EQ(inv_sqrt(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(-0.5 * inv_sqrt(0.5) / (0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

struct inv_sqrt_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return inv_sqrt(arg1);
  }
};

TEST(AgradFwdInvSqrt,inv_sqrt_NaN) {
  inv_sqrt_fun inv_sqrt_;
  test_nan_fwd(inv_sqrt_,false);
}
