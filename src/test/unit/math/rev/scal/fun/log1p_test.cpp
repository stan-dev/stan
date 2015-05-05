#include <stan/math/rev/scal/fun/log1p.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev,log1p) {
  AVAR a = 0.1;
  AVAR f = stan::math::log1p(a);
  EXPECT_FLOAT_EQ(log(1 + 0.1), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0 / (1.0 + 0.1), grad_f[0]);
}

struct log1p_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log1p(arg1);
  }
};

TEST(AgradRev,log1p_NaN) {
  log1p_fun log1p_;
  test_nan(log1p_,false,true);
}
