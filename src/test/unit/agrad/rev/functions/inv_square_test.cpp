#include <stan/agrad/rev/functions/inv_square.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/constants.hpp>
#include <test/unit/agrad/rev/nan_util.hpp>

TEST(AgradRev,inv_square) {
  AVAR a = 7.0;
  AVEC x = createAVEC(a);
  AVAR f = inv_square(a);
  EXPECT_FLOAT_EQ(1 / 49.0, f.val());

  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_EQ(1U,grad_f.size());
  EXPECT_FLOAT_EQ(-2.0 / 343.0, grad_f[0]);

  a = 0.0;
  x = createAVEC(a);
  f = inv_square(a);
  EXPECT_FLOAT_EQ(stan::math::positive_infinity(),f.val());

  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(stan::math::negative_infinity(),grad_f[0]);
}

struct inv_square_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return inv_square(arg1);
  }
};

TEST(AgradRev,inv_square_NaN) {
  inv_square_fun inv_square_;
  test_nan(inv_square_,false,true);
}
