#include <stan/agrad/rev/operators/operator_multiply_equal.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,a_timeseq_b) {
  AVAR a(5.0);
  AVAR b(-1.0);
  AVEC x = createAVEC(a,b);
  AVAR f = (a *= b);
  EXPECT_FLOAT_EQ(-5.0,f.val());
  EXPECT_FLOAT_EQ(-5.0,a.val());
  EXPECT_FLOAT_EQ(-1.0,b.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.0,g[0]);
  EXPECT_FLOAT_EQ(5.0,g[1]);
}

TEST(AgradRev,a_timeseq_bd) {
  AVAR a(5.0);
  double b = -1.0;
  AVEC x = createAVEC(a);
  AVAR f = (a *= b);
  EXPECT_FLOAT_EQ(-5.0,f.val());
  EXPECT_FLOAT_EQ(-5.0,a.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.0,g[0]);
}
