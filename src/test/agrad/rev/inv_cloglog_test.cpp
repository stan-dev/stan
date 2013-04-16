#include <stan/agrad/rev/inv_cloglog.hpp>
#include <stan/math/functions/inv_cloglog.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,inv_cloglog) {
  AVAR a = 2.7;
  AVAR f = inv_cloglog(a);
  EXPECT_FLOAT_EQ(stan::math::inv_cloglog(2.7),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::exp(2.7) * std::exp(-std::exp(2.7)),grad_f[0]);
}
