#include <gtest/gtest.h>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/trunc.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdTrunc, Fvar) {
  using stan::math::fvar;
  using boost::math::trunc;

  fvar<double> x(0.5,1.0);
  fvar<double> y(2.4,2.0);

  fvar<double> a = trunc(x);
  EXPECT_FLOAT_EQ(trunc(0.5), a.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_);

  fvar<double> b = trunc(y);
  EXPECT_FLOAT_EQ(trunc(2.4), b.val_);
  EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c = trunc(2 * x);
  EXPECT_FLOAT_EQ(trunc(2 * 0.5), c.val_);
  EXPECT_FLOAT_EQ(0.0, c.d_);
}


struct trunc_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return trunc(arg1);
  }
};

TEST(AgradFwdTrunc,trunc_NaN) {
  trunc_fun trunc_;
  test_nan_fwd(trunc_,false);
}
