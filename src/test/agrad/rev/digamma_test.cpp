#include <stan/agrad/rev/digamma.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/zeta.hpp>

TEST(AgradRev,digamma) {
  AVAR a = 3.5;
  AVAR f = digamma(a);
  EXPECT_FLOAT_EQ(boost::math::digamma(3.5),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(boost::math::zeta(2.0) - (0.57721566490153286 
                                       + boost::math::digamma(3.5)),grad_f[0]);
}  
