#include <stan/agrad/rev/operators/operator_unary_negative.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/rev/nan_util.hpp>

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
