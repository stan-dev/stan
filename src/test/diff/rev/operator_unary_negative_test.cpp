#include <stan/agrad/rev/operator_unary_negative.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,neg_a) {
  AVAR a = 5.0;
  AVAR f = -a;
  EXPECT_FLOAT_EQ(-5.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(-1.0,dx[0]);
}
