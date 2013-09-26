#include <stan/agrad/rev/cosh.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <cmath>

TEST(AgradRev,cosh_var) {
  AVAR a = 0.68;
  AVAR f = cosh(a);
  EXPECT_FLOAT_EQ(1.2402474, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(sinh(0.68), g[0]);
}

TEST(AgradRev,cosh_neg_var) {
  AVAR a = -.68;
  AVAR f = cosh(a);
  EXPECT_FLOAT_EQ(1.2402474,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(sinh(-.68),g[0]);
}

TEST(AgradRev,cosh_inf) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  AVAR f = cosh(a);
  EXPECT_FLOAT_EQ(inf,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_TRUE(std::isinf(g[0]) && (g[0] > 0));
}

TEST(AgradRev,cosh_neg_inf) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = -inf;
  AVAR f = cosh(a);
  EXPECT_FLOAT_EQ(inf,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_TRUE(std::isinf(g[0]) && (g[0] < 0));
}
