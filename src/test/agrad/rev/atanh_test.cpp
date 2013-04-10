#include <stan/agrad/rev/atanh.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,atanh) {
  AVAR a = 0.3;
  AVAR f = atanh(a);
  EXPECT_FLOAT_EQ(atanh(0.3), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0/(1.0 - 0.3 * 0.3), grad_f[0]);
}
