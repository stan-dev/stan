#include <stan/agrad/rev/atan.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,atan_var) {
  AVAR a = 0.68;
  AVAR f = atan(a);
  EXPECT_FLOAT_EQ(atan(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(1.0 + (0.68 * 0.68)), g[0]);
}
