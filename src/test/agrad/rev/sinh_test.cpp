#include <stan/agrad/rev/sinh.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,sinh_var) {
  AVAR a = 0.68;
  AVAR f = sinh(a);
  EXPECT_FLOAT_EQ(sinh(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(cosh(0.68),g[0]);
}
