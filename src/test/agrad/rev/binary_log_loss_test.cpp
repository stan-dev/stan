#include <stan/agrad/rev/binary_log_loss.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/functions/binary_log_loss.hpp>

TEST(AgradRev,binary_log_loss) {
  AVAR a = 0.4;
  AVAR f = stan::agrad::binary_log_loss(0,a);
  EXPECT_FLOAT_EQ(stan::math::binary_log_loss(0,0.4),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(2.5,grad_f[0]);
}
