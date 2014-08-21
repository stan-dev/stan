#include <stan/agrad/rev/functions/lmgamma.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/functions/lmgamma.hpp>
#include <test/unit-agrad-rev/nan_util.hpp>

TEST(AgradRev,lmgamma) {
  using stan::math::lmgamma;
  AVAR a = 3.2;
  AVAR f = stan::agrad::lmgamma(3,a);
  EXPECT_FLOAT_EQ(lmgamma(3,3.2),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(4.9138227 / 2.1,grad_f[0]);
}

struct lmgamma_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return lmgamma(3,arg1);
  }
};

TEST(AgradRev,lmgamma_NaN) {
  lmgamma_fun lmgamma_;
  test_nan(lmgamma_,false,true);
}
