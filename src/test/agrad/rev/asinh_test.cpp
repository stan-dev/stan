#include <stan/agrad/rev/asinh.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,asinh_val) {
  AVAR a = 0.2;
  AVAR f = asinh(a);
  EXPECT_FLOAT_EQ(0.198690110349, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/sqrt(0.2 * 0.2  + 1.0), g[0]);
}

TEST(AgradRev,asinh_neg_val) {
  AVAR a = -0.2;
  AVAR f = asinh(a);
  EXPECT_FLOAT_EQ(-0.198690110349, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/sqrt(-0.2 * -0.2  + 1.0), g[0]);
}

TEST(AgradRev,asinh_boundry) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  AVAR b = -inf;
  AVAR f = asinh(a);
  AVAR e = asinh(b);
  EXPECT_FLOAT_EQ(inf, f.val());
  EXPECT_FLOAT_EQ(-inf,e.val());
  
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ((0.0), g[0]);

  AVEC y = createAVEC(b);
  VEC h;
  f.grad(y,h);
  EXPECT_FLOAT_EQ((0.0), h[0]); 
}
