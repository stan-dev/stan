#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/agrad/rev/functions/log1m.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,log1m) {
  AVAR a = 0.1;
  AVAR f = log1m(a);
  EXPECT_FLOAT_EQ(log(1 - 0.1), f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-1.0/(1.0 - 0.1), grad_f[0]);
}
TEST(AgradRev,log1mErr) {
  AVAR a = 10;
  AVAR f = log1m(a);
  EXPECT_TRUE(boost::math::isnan(f.val()));
}

TEST(AgradRev,log1m_nan) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::log1m(a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
