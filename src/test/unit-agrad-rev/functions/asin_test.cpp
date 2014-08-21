#include <stan/agrad/rev/functions/asin.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/constants.hpp>
#include <stan/agrad/rev/numeric_limits.hpp>
#include <test/unit-agrad-rev/nan_util.hpp>

TEST(AgradRev,asin_var) {
  AVAR a = 0.68;
  AVAR f = asin(a);
  EXPECT_FLOAT_EQ(asin(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/sqrt(1.0 - (0.68 * 0.68)), g[0]);
}

TEST(AgradRev,asin_1) {
  AVAR a = 1;
  AVAR f = asin(a);
  EXPECT_FLOAT_EQ((1.57079632679),f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/sqrt(1.0 - (1*1)),g[0]);
}

TEST(AgradRev,asin_neg_1) {
  AVAR a = -1;
  AVAR f = asin(a);
  EXPECT_FLOAT_EQ((-1.57079632679),f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/sqrt(1.0 - (-1*-1)),g[0]);
}

TEST(AgradRev,asin_out_of_bounds) {
  AVAR a = 1.0 + stan::math::EPSILON;
  EXPECT_TRUE(std::isnan(asin(a)));

  a = -1.0 - stan::math::EPSILON;
  EXPECT_TRUE(std::isnan(asin(a)));
}

TEST(AgradRev,asin_nan) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::asin(a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}

struct asin_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return asin(arg1);
  }
};

TEST(AgradRev,asin_NaN) {
  asin_fun asin_;
  test_nan(asin_,false,true);
}
