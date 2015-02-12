#include <stan/math/rev/scal/fun/ceil.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev,ceil_var) {
  AVAR a = 1.9;
  AVAR f = ceil(a);
  EXPECT_FLOAT_EQ(2.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

struct ceil_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return ceil(arg1);
  }
};

TEST(AgradRev,ceil_NaN) {
  ceil_fun ceil_;
  test_nan(ceil_,false,true);
}
