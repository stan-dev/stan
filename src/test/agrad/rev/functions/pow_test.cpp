#include <stan/agrad/rev/functions/pow.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,pow_var_var) {
  AVAR a(3.0);
  AVAR b(4.0);
  AVAR f = pow(a,b);
  EXPECT_FLOAT_EQ(81.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(4.0 * pow(3.0,4.0-1.0), g[0]);
  EXPECT_FLOAT_EQ(log(3.0) * pow(3.0,4.0), g[1]);
}

TEST(AgradRev,pow_var_double) {
  AVAR a(3.0);
  double b = 4.0;
  AVAR f = pow(a,b);
  EXPECT_FLOAT_EQ(81.0,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(4.0 * pow(3.0,4.0-1.0), g[0]);
}


TEST(AgradRev,pow_double_var) {
  double a = 3.0;
  AVAR b(4.0);
  AVAR f = pow(a,b);
  EXPECT_FLOAT_EQ(81.0,f.val());

  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(log(3.0) * pow(3.0,4.0), g[0]);
}

TEST(AgradRev,pow_boundry) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  AVAR b = 5;
  AVAR f = pow(a,b);
  EXPECT_FLOAT_EQ(inf, f.val());
  AVAR g = pow(b,a);
  EXPECT_FLOAT_EQ(inf, g.val());

  AVAR c = -inf;
  AVAR d = 6;
  AVAR h = pow(c,b);
  EXPECT_FLOAT_EQ(-inf, h.val());
  AVAR i = pow(c,d);
  EXPECT_FLOAT_EQ(inf, i.val());

  AVAR j = pow(b,c);
  EXPECT_FLOAT_EQ( 0.0 ,j.val());
}
