#include <stan/agrad/rev/functions/exp.hpp>
#include <test/unit/agrad/util.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,exp_a) {
  AVAR a(6.0);
  AVAR f = exp(a); // mix exp() functs w/o namespace
  EXPECT_FLOAT_EQ(exp(6.0),f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(exp(6.0),g[0]);
}

TEST(AgradRev,exp_nan) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::exp(a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
