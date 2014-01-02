#include <stan/agrad/rev/functions/tan.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/constants.hpp>
#include <stan/agrad/rev/numeric_limits.hpp>

TEST(AgradRev,tan_var) {
  AVAR a = 0.68;
  AVAR f = tan(a);
  EXPECT_FLOAT_EQ(0.80866137, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1 + tan(0.68)*tan(0.68), g[0]);
}

TEST(AgradRev,tan_neg_var) {
  AVAR a = -.68;
  AVAR f = tan(a);
  EXPECT_FLOAT_EQ(-0.80866137, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1 + tan(-0.68)*tan(-0.68), g[0]);
}

TEST(AgradRev,tan_boundry) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  EXPECT_TRUE(std::isnan(tan(a)))
    << "tan(" << a << "): " << tan(a) << " mimics std::tan behavior";

  AVAR b = -inf;
  EXPECT_TRUE(std::isnan(tan(b)))
    << "tan(" << b << "): " << tan(b) << " mimics std::tan behavior";
}

