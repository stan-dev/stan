#include <stan/diff/rev/fabs.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,fabs_var) {
  AVAR a = 0.68;
  AVAR f = fabs(a);
  EXPECT_FLOAT_EQ(0.68, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}

TEST(DiffRev,fabs_var_2) {
  AVAR a = -0.68;
  AVAR f = fabs(a);
  EXPECT_FLOAT_EQ(0.68, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.0, g[0]);
}

TEST(DiffRev,fabs_var_3) {
  AVAR a = 0.0;
  AVAR f = fabs(a);
  EXPECT_FLOAT_EQ(0.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}
