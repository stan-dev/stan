#include <stan/agrad/rev/functions/hypot.hpp>
#include <test/unit/agrad/util.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,hypot_vv) {
  AVAR a = 3.0;
  AVAR b = 4.0;
  AVAR f = hypot(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but doc this way
  EXPECT_FLOAT_EQ(3.0/5.0,grad_f[0]);
  EXPECT_FLOAT_EQ(4.0/5.0,grad_f[1]);
}  

TEST(AgradRev,hypot_vd) {
  AVAR a = 3.0;
  double b = 4.0;
  AVAR f = hypot(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but doc this way
  EXPECT_FLOAT_EQ(3.0/5.0,grad_f[0]);
}  

TEST(AgradRev,hypot_dv) {
  double a = 3.0;
  AVAR b = 4.0;
  AVAR f = hypot(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but doc this way
  EXPECT_FLOAT_EQ(4.0/5.0,grad_f[0]);
}  

TEST(AgradRev,hypot_nan_vv) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR b = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::hypot(a,b);

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(2U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
  EXPECT_TRUE(boost::math::isnan(g[1]));
}

TEST(AgradRev,hypot_nan_vd) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::hypot(a,1);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}

TEST(AgradRev,hypot_nan_dv) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::hypot(1,a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
