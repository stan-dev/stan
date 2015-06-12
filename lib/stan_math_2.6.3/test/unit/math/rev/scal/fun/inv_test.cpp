#include <stan/math/rev/scal/fun/inv.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev,inv) {
  AVAR a = 7.0;
  AVEC x = createAVEC(a);
  AVAR f = inv(a);
  EXPECT_FLOAT_EQ(1 / 7.0, f.val());

  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_EQ(1U,grad_f.size());
  EXPECT_FLOAT_EQ(-1 / 49.0, grad_f[0]);

  a = 0.0;
  x = createAVEC(a);
  f = inv(a);
  EXPECT_FLOAT_EQ(stan::math::positive_infinity(),f.val());

  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(stan::math::negative_infinity(),grad_f[0]);
}

struct inv_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return inv(arg1);
  }
};

TEST(AgradRev,inv_NaN) {
  inv_fun inv_;
  test_nan(inv_,false,true);
}
