#include <stan/diff/rev/ceil.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,ceil_var) {
  AVAR a = 1.9;
  AVAR f = ceil(a);
  EXPECT_FLOAT_EQ(2.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}
