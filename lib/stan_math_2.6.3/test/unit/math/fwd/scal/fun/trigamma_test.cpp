#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/trigamma.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>

TEST(AgradFwdTrigamma, Fvar) {
  using stan::math::fvar;
  using stan::math::trigamma;

  fvar<double> x(0.5,1.0);
  fvar<double> a = trigamma(x);
  EXPECT_FLOAT_EQ(4.9348022005446793094, a.val_);
  EXPECT_FLOAT_EQ(-16.8288, a.d_);
}

struct trigamma_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return stan::math::trigamma(arg1);
  }
};

TEST(AgradFwdTrigamma,trigamma_NaN) {
  trigamma_fun trigamma_;
  test_nan_fwd(trigamma_,false);
}
