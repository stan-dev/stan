#include <limits>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/agrad/rev/abs.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,abs_var) {
  AVAR a = 0.68;
  AVAR f = abs(a);
  EXPECT_FLOAT_EQ(0.68, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}

TEST(AgradRev,abs_var_2) {
  AVAR a = -0.68;
  AVAR f = abs(a);
  EXPECT_FLOAT_EQ(0.68, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.0, g[0]);
}

TEST(AgradRev,abs_var_3) {
  AVAR a = 0.0;
  AVAR f = abs(a);
  EXPECT_FLOAT_EQ(0.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_EQ(1,g.size());
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

TEST(AgradRev,abs_NaN) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = abs(a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
