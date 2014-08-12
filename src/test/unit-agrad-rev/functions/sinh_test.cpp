#include <stan/agrad/rev/functions/sinh.hpp>
#include <test/unit/agrad/util.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,sinh_var) {
  AVAR a = 0.68;
  AVAR f = sinh(a);
  EXPECT_FLOAT_EQ(0.73363036, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(cosh(0.68),g[0]);
}

TEST(AgradRev,sinh_neg_var) {
  AVAR a = -.68;
  AVAR f = sinh(a);
  EXPECT_FLOAT_EQ(-0.73363036,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(cosh(-.68),g[0]);
}

TEST(AgradRev,sinh_inf) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  AVAR f = sinh(a);
  EXPECT_FLOAT_EQ(inf,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(inf,g[0]);
}

TEST(AgradRev,sinh_neg_inf) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = -inf;
  AVAR f = sinh(a);
  EXPECT_FLOAT_EQ(-inf,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(inf,g[0]);
}

TEST(AgradRev,sinh_nan) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::sinh(a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
