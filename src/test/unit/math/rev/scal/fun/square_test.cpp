#include <stan/math/rev/scal/fun/square.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev,square) {
  AVAR a = 7.0;
  AVEC x = createAVEC(a);
  AVAR f = square(a);
  EXPECT_FLOAT_EQ(49.0, f.val());

  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_EQ(1U,grad_f.size());
  EXPECT_FLOAT_EQ(14.0, grad_f[0]);
}

struct square_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return square(arg1);
  }
};

TEST(AgradRev,square_NaN) {
  square_fun square_;
  test_nan(square_,false,true);
}
