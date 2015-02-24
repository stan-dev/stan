#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev,neg_a) {
  AVAR a = 5.0;
  AVAR f = -a;
  EXPECT_FLOAT_EQ(-5.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(-1.0,dx[0]);
}

struct neg_fun {
  template <typename T0>
  inline T0
  operator()(T0 arg1) const {
    return (-arg1);
  }
};

TEST(AgradRev, neg_nan) {
  neg_fun neg_;

  test_nan(neg_,false, true);
}
