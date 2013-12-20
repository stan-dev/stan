#include <stan/agrad/rev/functions/erf.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,erf) {
  AVAR a = 1.3;
  AVAR f = erf(a);
  EXPECT_FLOAT_EQ(boost::math::erf(1.3), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(2.0 / std::sqrt(boost::math::constants::pi<double>()) * std::exp(- 1.3 * 1.3), grad_f[0]);
}
