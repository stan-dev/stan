#include <stan/agrad/rev/functions/lgamma.hpp>
#include <test/unit/agrad/util.hpp>
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

TEST(AgradRev,lgamma_nan) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::lgamma(a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
