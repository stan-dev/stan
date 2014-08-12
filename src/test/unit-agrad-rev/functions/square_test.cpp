#include <stan/agrad/rev/functions/square.hpp>
#include <test/unit/agrad/util.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,square) {
  AVAR a = 7.0;
  AVEC x = createAVEC(a);
  AVAR f = square(a);
  EXPECT_FLOAT_EQ(49.0, f.val());

  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_EQ(1U,grad_f.size());
  EXPECT_FLOAT_EQ(14.0, grad_f[0]);
}

TEST(AgradRev,square_nan) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::square(a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
