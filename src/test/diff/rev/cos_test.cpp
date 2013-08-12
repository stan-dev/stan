#include <stan/diff/rev/cos.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,cos_var) {
  AVAR a = 0.43;
  AVAR f = cos(a);
  EXPECT_FLOAT_EQ(cos(0.43), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-sin(0.43),g[0]);
}
