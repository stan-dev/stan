#include <stan/agrad/rev/operators/operator_divide_equal.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,a_divideeq_b) {
  AVAR a(6.0);
  AVAR b(-2.0);
  AVEC x = createAVEC(a,b);
  AVAR f = (a /= b);
  EXPECT_FLOAT_EQ(-3.0,f.val());
  EXPECT_FLOAT_EQ(-3.0,a.val());
  EXPECT_FLOAT_EQ(-2.0,b.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/-2.0,g[0]);
  EXPECT_FLOAT_EQ(-6.0/((-2.0)*(-2.0)),g[1]);
}

TEST(AgradRev,a_divideeq_bd) {
  AVAR a(6.0);
  double b = -2.0;
  AVEC x = createAVEC(a);
  AVAR f = (a /= b);
  EXPECT_FLOAT_EQ(-3.0,f.val());
  EXPECT_FLOAT_EQ(-3.0,a.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/-2.0,g[0]);
}
