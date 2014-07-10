#include <limits>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/agrad/rev/functions/fabs.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,fabs_var) {
  AVAR a = 0.68;
  AVAR f = fabs(a);
  EXPECT_FLOAT_EQ(0.68, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}

TEST(AgradRev,fabs_var_2) {
  AVAR a = -0.68;
  AVAR f = fabs(a);
  EXPECT_FLOAT_EQ(0.68, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.0, g[0]);
}

TEST(AgradRev,fabs_var_3) {
  AVAR a = 0.0;
  AVAR f = fabs(a);
  EXPECT_FLOAT_EQ(0.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

TEST(AgradRev, fabs_NaN) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = fabs(a);

  
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_TRUE(boost::math::isnan(f.val()));
  EXPECT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
