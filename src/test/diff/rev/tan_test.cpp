#include <stan/diff/rev/tan.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,tan_var) {
  AVAR a = 0.68;
  AVAR f = tan(a);
  EXPECT_FLOAT_EQ(tan(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1 + tan(0.68)*tan(0.68), g[0]);
}
