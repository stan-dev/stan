#include <stan/agrad/rev/log1p.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,log1p) {
  AVAR a = 0.1;
  AVAR f = stan::agrad::log1p(a);
  EXPECT_FLOAT_EQ(log(1 + 0.1), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0 / (1.0 + 0.1), grad_f[0]);
}
