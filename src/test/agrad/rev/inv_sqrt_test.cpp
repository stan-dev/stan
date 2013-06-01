#include <stan/agrad/rev/inv_sqrt.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/constants.hpp>

TEST(AgradRev,inv_sqrt) {
  AVAR a = 49.0;
  AVEC x = createAVEC(a);
  AVAR f = inv_sqrt(a);
  EXPECT_FLOAT_EQ(1 / 7.0, f.val());

  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_EQ(1U,grad_f.size());
  EXPECT_FLOAT_EQ(-0.5 / (7 * 49), grad_f[0]);

  a = 0.0;
  x = createAVEC(a);
  f = inv_sqrt(a);
  EXPECT_FLOAT_EQ(stan::math::positive_infinity(),f.val());

  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(stan::math::negative_infinity(),grad_f[0]);

  a = -50.0;
  x = createAVEC(a);
  f = inv_sqrt(a);
  std::isnan(f.val());

  f.grad(x,grad_f);
  std::isnan(grad_f[0]);
}
