#include <stan/agrad/rev/sin.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/rev/numeric_limits.hpp>

TEST(AgradRev,sin_var) {
  AVAR a = 0.49;
  AVAR f = sin(a);
  EXPECT_FLOAT_EQ((.470625888), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(cos(0.49),g[0]);
}

TEST(AgradRev,sin_neg_var) {
  AVAR a = -0.49;
  AVAR f = sin(a);
  EXPECT_FLOAT_EQ((-.470625888), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(cos(-0.49), g[0]);
}

TEST(AgradRev,sin_boundry) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  EXPECT_TRUE(std::isnan(sin(a)));

  AVAR b = -inf;
  EXPECT_TRUE(std::isnan(sin(b)));
}
