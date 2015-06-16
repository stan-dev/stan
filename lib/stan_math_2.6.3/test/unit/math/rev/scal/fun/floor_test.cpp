#include <stan/math/rev/scal/fun/floor.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev,floor_var) {
  AVAR a = 1.2;
  AVAR f = floor(a);
  EXPECT_FLOAT_EQ(1.0, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
}

struct floor_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return floor(arg1);
  }
};

TEST(AgradRev,floor_NaN) {
  floor_fun floor_;
  test_nan(floor_,false,true);
}
