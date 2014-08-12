#include <stan/agrad/rev/functions/atan2.hpp>
#include <stan/agrad/rev/operators/operator_division.hpp>
#include <stan/agrad/rev/operators/operator_multiplication.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,atan2_var_var) {
  AVAR a = 1.2;
  AVAR b = 3.9;
  AVAR f = atan2(a,b);
  EXPECT_FLOAT_EQ(atan2(1.2,3.9),f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(3.9 / (1.2 * 1.2 + 3.9 * 3.9), g[0]);
  EXPECT_FLOAT_EQ(-1.2 / (1.2 * 1.2 + 3.9 * 3.9), g[1]);
}

TEST(AgradRev,atan2_dvd) {
  AVAR sigma = 1;
  AVEC x = createAVEC(sigma);
  AVAR f = atan2(1.0,sigma) / 3.14;
  VEC g;
  f.grad(x,g);

  AVAR sigma1 = 1;
  AVEC x1 = createAVEC(sigma1);
  AVAR f1 = atan2(1.0,sigma1);
  VEC g1;
  f1.grad(x1,g1);

  EXPECT_FLOAT_EQ(3.14 * g[0],g1[0]);
}
TEST(AgradRev,atan2_var_var__integration) {
  double c = 5.0;
  AVAR a = 1.2;
  AVAR b = 3.9;
  AVAR f = atan2(a,b) * c;
  EXPECT_FLOAT_EQ(atan2(1.2,3.9)*c,f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(3.9 / (1.2 * 1.2 + 3.9 * 3.9) * c, g[0]);
  EXPECT_FLOAT_EQ(-1.2 / (1.2 * 1.2 + 3.9 * 3.9) * c, g[1]);
}


TEST(AgradRev,atan2_var_double) {
  AVAR a = 1.2;

  double b = 3.9;
  AVAR f = atan2(a,b);
  EXPECT_FLOAT_EQ(atan2(1.2,3.9),f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(3.9 / (1.2 * 1.2 + 3.9 * 3.9), g[0]);
}

TEST(AgradRev,atan2_double_var) {
  double a = 1.2;
  AVAR b = 3.9;
  AVAR f = atan2(a,b);
  EXPECT_FLOAT_EQ(atan2(1.2,3.9),f.val());

  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.2 / (1.2 * 1.2 + 3.9 * 3.9), g[0]);
}

TEST(AgradRev,atan2_nan_vv) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR b = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::atan2(a,b);

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(2U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
  EXPECT_TRUE(boost::math::isnan(g[1]));
}

TEST(AgradRev,atan2_nan_vd) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::atan2(a,1);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}

TEST(AgradRev,atan2_nan_dv) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::atan2(1,a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
