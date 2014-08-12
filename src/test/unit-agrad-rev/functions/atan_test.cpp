#include <stan/agrad/rev/functions/atan.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <test/unit/agrad/util.hpp>
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
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);

  AVAR b = -inf;
  AVAR e = atan(b);
  EXPECT_FLOAT_EQ(-1.5707964,e.val());
  AVEC y = createAVEC(b);
  VEC h;
  e.grad(y, h);
  EXPECT_FLOAT_EQ(0.0, h[0]);
}

TEST(AgradRev,atan_nan) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::atan(a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
