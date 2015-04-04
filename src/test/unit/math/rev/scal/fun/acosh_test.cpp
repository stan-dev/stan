#include <stan/math/rev/scal/fun/acosh.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev,acosh_val) {
  AVAR a = 1.3;
  AVAR f = acosh(a);
  EXPECT_FLOAT_EQ(acosh(1.3), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/sqrt(1.3 * 1.3  - 1.0), g[0]);
}

TEST(AgradRev,acosh_1) {
  AVAR a = 1;
  AVAR f = acosh(a);
  EXPECT_FLOAT_EQ(0.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/sqrt(-1.0 * -1.0 - 1.0), g[0]);
}

TEST(AgradRev,acosh_inf) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  AVAR f = acosh(a);
  EXPECT_FLOAT_EQ(inf, f.val());
  
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ((0.0), g[0]);
}

TEST(AgradRev,acosh_out_of_bounds) {
  AVAR a = 1.0 - stan::math::EPSILON;
  AVAR f = acosh(a);
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);

  EXPECT_TRUE(std::isnan(acosh(a)));
  EXPECT_TRUE(g.size() == 1);
  EXPECT_TRUE(std::isnan(g[0]));

  AVAR b = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isinf(acosh(b)) && acosh(b) > 0);
}

struct acosh_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return acosh(arg1);
  }
};

TEST(AgradRev,acosh_NaN) {
  acosh_fun acosh_;
  test_nan(acosh_,false,true);
}
