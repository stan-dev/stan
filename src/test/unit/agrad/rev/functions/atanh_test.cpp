#include <stan/agrad/rev/functions/atanh.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/constants.hpp>
#include <test/unit/agrad/rev/nan_util.hpp>

TEST(AgradRev,atanh) {
  AVAR a = 0.3;
  AVAR f = atanh(a);
  EXPECT_FLOAT_EQ(atanh(0.3), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(1.0 - 0.3 * 0.3), g[0]);
}

TEST(AgradRev,atanh_1) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = 1;
  AVAR f = atanh(a);
  EXPECT_FLOAT_EQ(inf, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(1.0 - 1.0 * 1.0), g[0]);
}

TEST(AgradRev,atanh_neg_1) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = -1;
  AVAR f = atanh(a);
  EXPECT_FLOAT_EQ(-inf, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/(1.0 - (-1.0 * -1.0)), g[0]);
}

TEST(AgradRev,atanh_out_of_bounds) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a =  1.0 + stan::math::EPSILON;
  AVAR b = -1.0 - stan::math::EPSILON;
  AVAR c =  inf;
  AVAR d = -inf;
  EXPECT_TRUE(boost::math::isnan(atanh(a).val()));
  EXPECT_TRUE(boost::math::isnan(atanh(b).val()));
  EXPECT_TRUE(boost::math::isnan(atanh(c).val()));
  EXPECT_TRUE(boost::math::isnan(atanh(d).val()));
}

struct atanh_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return atanh(arg1);
  }
};

TEST(AgradRev,atanh_NaN) {
  atanh_fun atanh_;
  test_nan(atanh_,false,true);
}
