#include <stan/agrad/rev/functions/log10.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,log10_a) {
  AVAR a(5.0);
  AVAR f = log10(a); 
  EXPECT_FLOAT_EQ(log10(5.0),f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(log(10.0) * 5.0),g[0]);
}

TEST(AgradRev,log10_nan) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::log10(a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
