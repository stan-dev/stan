#include <stan/agrad/rev/functions/digamma.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/zeta.hpp>

TEST(AgradRev,digamma) {
  AVAR a = 0.5;
  AVAR f = digamma(a);
  EXPECT_FLOAT_EQ(boost::math::digamma(0.5),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(4.9348022005446793094, grad_f[0]);
}  
