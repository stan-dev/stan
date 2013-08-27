#include <stan/diff/rev/sin.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,sin_var) {
  AVAR a = 0.49;
  AVAR f = sin(a);
  EXPECT_FLOAT_EQ(sin(0.49), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
   EXPECT_FLOAT_EQ(cos(0.49),g[0]);
}
