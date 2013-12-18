#include <stan/agrad/rev/functions/cos.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/rev/numeric_limits.hpp>

TEST(AgradRev,cos_var) {
  AVAR a = 0.49;
  AVAR f = cos(a);
  EXPECT_FLOAT_EQ(.8823329, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-sin(0.49),g[0]);
}

TEST(AgradRev,cos_neg_var) {
  AVAR a = -0.49;
  AVAR f = cos(a);
  EXPECT_FLOAT_EQ((.8823329), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-sin(-0.49), g[0]);
}

TEST(AgradRev,cos_boundry) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  EXPECT_TRUE(std::isnan(cos(a)));

  AVAR b = -inf;
  EXPECT_TRUE(std::isnan(cos(b)));
}
