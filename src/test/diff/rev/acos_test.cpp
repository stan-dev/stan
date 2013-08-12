#include <stan/agrad/rev/acos.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,acos_var) {
  AVAR a = 0.68;
  AVAR f = acos(a);
  EXPECT_FLOAT_EQ(acos(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.0/sqrt(1.0 - (0.68 * 0.68)), g[0]);
}
