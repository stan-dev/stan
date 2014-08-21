#include <stan/agrad/rev/operators/operator_unary_plus.hpp>
#include <stan/agrad/rev/operators/operator_equal.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit-agrad-rev/nan_util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,pos_a) {
  AVAR a = 5.0;
  AVAR f = +a;
  EXPECT_FLOAT_EQ(5.0,f.val());
  EXPECT_TRUE(a == +a);
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
}

struct pos_fun {
  template <typename T0>
  inline T0
  operator()(T0 arg1) const {
    return (+arg1);
  }
};

TEST(AgradRev, pos_nan) {
  pos_fun pos_;

  test_nan(pos_,false, false);
}
