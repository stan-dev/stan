#include <stan/agrad/rev/tgamma.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

TEST(AgradRev,tgamma) {
  AVAR a = 3.5;
  AVAR f = tgamma(a);
  EXPECT_FLOAT_EQ(boost::math::tgamma(3.5),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(boost::math::digamma(3.5) * boost::math::tgamma(3.5),grad_f[0]);
}  
