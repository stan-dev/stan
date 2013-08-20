#include <stan/agrad/rev/atan.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,atan_1) {
  AVAR a = 1;
  AVAR f = atan(a);
  EXPECT_FLOAT_EQ((.78539816339),f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(1.0 + (1 * 1)),g[0]);
}

TEST(AgradRev,atan_neg_1) {
  AVAR a = -1;
  AVAR f = atan(a);
  EXPECT_FLOAT_EQ((-.78539816339),f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(1.0 + (-1*-1)),g[0]);
}

TEST(AgradRev,atan_boundry) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  AVAR f = atan(a);
  EXPECT_FLOAT_EQ(1.5707964,f.val());

  AVAR b = -inf;
  AVAR e = atan(b);
  EXPECT_FLOAT_EQ(-1.5707964,e.val());
}
