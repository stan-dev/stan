#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/functions/trigamma.hpp>
#include <stan/agrad/rev.hpp>

TEST(AgradRev,trigamma) {
  AVAR a = 0.5;
  AVAR f = stan::math::trigamma(a);
  EXPECT_FLOAT_EQ(4.9348022005446793094,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-16.8288, grad_f[0]);
}  

TEST(AgradRev,trigamma_nan) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::math::trigamma(a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
