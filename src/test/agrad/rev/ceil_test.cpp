#include <stan/agrad/rev/ceil.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,ceil_var) {
  AVAR a = 1.9;
  AVAR f = ceil(a);
  EXPECT_FLOAT_EQ(2.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}
