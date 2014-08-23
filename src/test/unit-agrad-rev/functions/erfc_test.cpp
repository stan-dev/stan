#include <stan/agrad/rev/functions/erfc.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,erfc) {
  AVAR a = 1.3;
  AVAR f = erfc(a);
  EXPECT_FLOAT_EQ(boost::math::erfc(1.3), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-2.0 / std::sqrt(boost::math::constants::pi<double>()) * std::exp(- 1.3 * 1.3), grad_f[0]);
}

TEST(AgradRev,erfc_nan) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::erfc(a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
