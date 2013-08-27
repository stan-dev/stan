#include <stan/diff/rev/cbrt.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,cbrt) {
  AVAR a = 27.0;
  AVAR f = cbrt(a);
  EXPECT_FLOAT_EQ(3.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0 / 3.0 / std::pow(27.0,2.0/3.0), grad_f[0]);
}
