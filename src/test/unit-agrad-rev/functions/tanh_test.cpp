#include <stan/agrad/rev/functions/tanh.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(AgradRev,tanh_var) {
  AVAR a = 0.68;
  AVAR f = tanh(a);
  EXPECT_FLOAT_EQ(0.59151939543, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(cosh(0.68) * cosh(0.68)), g[0]);
}

TEST(AgradRev,tanh_neg_var) {
  AVAR a = -.68;
  AVAR f = tanh(a);
  EXPECT_FLOAT_EQ(-0.59151939543,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(cosh(-0.68) * cosh(-0.68)), g[0]);
}

TEST(AgradRev,sinh_inf) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  AVAR f = tanh(a);
  EXPECT_FLOAT_EQ(1.0,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(cosh(inf) * cosh(inf)), g[0]);
}

TEST(AgradRev,sinh_neg_inf) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = -inf;
  AVAR f = tanh(a);
  EXPECT_FLOAT_EQ(-1,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(cosh(-inf) * cosh(-inf)), g[0]);
}

TEST(AgradRev,tanh_nan) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::tanh(a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
