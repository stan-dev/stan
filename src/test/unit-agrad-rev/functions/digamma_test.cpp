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

TEST(AgradRev,digamma_nan) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::digamma(a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
