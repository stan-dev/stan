#include <stan/agrad/rev/functions/lgamma.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>

TEST(AgradRev,lgamma) {
  AVAR a = 3.0;
  AVAR f = lgamma(a);
  EXPECT_FLOAT_EQ(lgamma(3.0),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(boost::math::digamma(3.0),grad_f[0]);
}
