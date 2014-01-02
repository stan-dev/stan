#include <stan/agrad/rev/functions/log10.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,log10_a) {
  AVAR a(5.0);
  AVAR f = log10(a); 
  EXPECT_FLOAT_EQ(log10(5.0),f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(log(10.0) * 5.0),g[0]);
}
