#include <stan/agrad/rev/functions/inv_logit.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,inv_logit) {
  AVAR a = 2.0;
  AVAR f = inv_logit(a);
  EXPECT_FLOAT_EQ(1.0 / (1.0 + exp(-2.0)),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(exp(-2.0)/pow(1 + exp(-2.0),2.0),
                  grad_f[0]);
}

TEST(AgradRev,inv_logit_nan) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::inv_logit(a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
