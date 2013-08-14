#include <stan/diff/rev/tanh.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,tanh_var) {
  AVAR a = 0.68;
  AVAR f = tanh(a);
  EXPECT_FLOAT_EQ(tanh(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(cosh(0.68) * cosh(0.68)), g[0]);
}
