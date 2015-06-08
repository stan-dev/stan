#include <stan/math/rev/scal/fun/tan.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev,tan_var) {
  AVAR a = 0.68;
  AVAR f = tan(a);
  EXPECT_FLOAT_EQ(0.80866137, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1 + tan(0.68)*tan(0.68), g[0]);
}

TEST(AgradRev,tan_neg_var) {
  AVAR a = -.68;
  AVAR f = tan(a);
  EXPECT_FLOAT_EQ(-0.80866137, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1 + tan(-0.68)*tan(-0.68), g[0]);
}

TEST(AgradRev,tan_boundry) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  EXPECT_TRUE(std::isnan(tan(a)))
    << "tan(" << a << "): " << tan(a) << " mimics std::tan behavior";

  AVAR b = -inf;
  EXPECT_TRUE(std::isnan(tan(b)))
    << "tan(" << b << "): " << tan(b) << " mimics std::tan behavior";
}

struct tan_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return tan(arg1);
  }
};

TEST(AgradRev,tan_NaN) {
  tan_fun tan_;
  test_nan(tan_,false,true);
}
