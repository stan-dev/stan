#include <stan/agrad/rev/inv_sqrt.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,inv_sqrt) {
  AVAR a = 49.0;
  AVEC x = createAVEC(a);
  AVAR f = inv_sqrt(a);
  EXPECT_FLOAT_EQ(1 / 7.0, f.val());

  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_EQ(1U,grad_f.size());
  EXPECT_FLOAT_EQ(-0.5 / (7 * 49), grad_f[0]);
}
