#include <stan/agrad/rev/acosh.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,acosh) {
  AVAR a = 1.3;
  AVAR f = acosh(a);
  EXPECT_FLOAT_EQ(acosh(1.3), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0/sqrt(1.3 * 1.3  - 1.0), grad_f[0]);
}
