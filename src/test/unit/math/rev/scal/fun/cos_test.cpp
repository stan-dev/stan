#include <stan/math/rev/scal/fun/cos.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev,cos_var) {
  AVAR a = 0.49;
  AVAR f = cos(a);
  EXPECT_FLOAT_EQ(.8823329, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-sin(0.49),g[0]);
}

TEST(AgradRev,cos_neg_var) {
  AVAR a = -0.49;
  AVAR f = cos(a);
  EXPECT_FLOAT_EQ((.8823329), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-sin(-0.49), g[0]);
}

TEST(AgradRev,cos_boundry) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  EXPECT_TRUE(std::isnan(cos(a)));

  AVAR b = -inf;
  EXPECT_TRUE(std::isnan(cos(b)));
}

struct cos_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return cos(arg1);
  }
};

TEST(AgradRev,cos_NaN) {
  cos_fun cos_;
  test_nan(cos_,false,true);
}
