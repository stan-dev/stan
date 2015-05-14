#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/inv.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/inv.hpp>

TEST(AgradFwdInv,Fvar) {
  using stan::math::fvar;
  using stan::math::inv;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // Derivatives w.r.t. x
  fvar<double> a = inv(x);

  EXPECT_FLOAT_EQ(inv(0.5), a.val_);
  EXPECT_FLOAT_EQ(-1 / 0.25, a.d_);

  fvar<double> b = 3 * inv(x) + x;
  EXPECT_FLOAT_EQ(3 * inv(0.5) + 0.5, b.val_);
  EXPECT_FLOAT_EQ(-3 / 0.25 + 1, b.d_);

  fvar<double> c = -inv(x) + 5;
  EXPECT_FLOAT_EQ(-inv(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(1 / 0.25, c.d_);

  fvar<double> d = -3 * inv(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * inv(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(3 / 0.25 + 5, d.d_);

  fvar<double> e = -3 * inv(-x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * inv(-0.5) + 5 * 0.5, e.val_);
  EXPECT_FLOAT_EQ(-3 / 0.25 + 5, e.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> f = inv(y);
  EXPECT_FLOAT_EQ(inv(-0.5), f.val_);
  EXPECT_FLOAT_EQ(-1 / 0.25, f.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> g = inv(z);
  EXPECT_FLOAT_EQ(stan::math::positive_infinity(), g.val_);
  EXPECT_FLOAT_EQ(stan::math::negative_infinity(), g.d_);
}   

struct inv_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return inv(arg1);
  }
};

TEST(AgradFwdInv,inv_NaN) {
  inv_fun inv_;
  test_nan_fwd(inv_,false);
}
