#include <stan/agrad/rev/functions/exp.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,exp_a) {
  AVAR a(6.0);
  AVAR f = exp(a); // mix exp() functs w/o namespace
  EXPECT_FLOAT_EQ(exp(6.0),f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(exp(6.0),g[0]);
}
