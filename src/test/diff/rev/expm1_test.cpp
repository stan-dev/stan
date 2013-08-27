#include <stan/diff/rev/expm1.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/expm1.hpp>

TEST(DiffRev,expm1) {
  AVAR a = 1.3;
  AVAR f = expm1(a);
  EXPECT_FLOAT_EQ(boost::math::expm1(1.3), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::exp(1.3) - 1.0, grad_f[0]);
}  
