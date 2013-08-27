#include <stan/diff/rev/inv_logit.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,inv_logit) {
  AVAR a = 2.0;
  AVAR f = inv_logit(a);
  EXPECT_FLOAT_EQ(1.0 / (1.0 + exp(-2.0)),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(exp(-2.0)/pow(1 + exp(-2.0),2.0),
                  grad_f[0]);
}
