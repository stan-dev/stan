#include <stan/math/rev/scal/fun/asinh.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev,asinh_val) {
  AVAR a = 0.2;
  AVAR f = asinh(a);
  EXPECT_FLOAT_EQ(0.198690110349, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/sqrt(0.2 * 0.2  + 1.0), g[0]);
}

TEST(AgradRev,asinh_neg_val) {
  AVAR a = -0.2;
  AVAR f = asinh(a);
  EXPECT_FLOAT_EQ(-0.198690110349, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/sqrt(-0.2 * -0.2  + 1.0), g[0]);
}

TEST(AgradRev,asinh_boundry) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  AVAR f = asinh(a);
  EXPECT_FLOAT_EQ(inf, f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);
  
  AVAR b = -inf;
  AVAR e = asinh(b);
  EXPECT_FLOAT_EQ(-inf,e.val());
  AVEC y = createAVEC(b);
  VEC h;
  e.grad(y,h);
  EXPECT_FLOAT_EQ(0.0, h[0]); 
}

struct asinh_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return asinh(arg1);
  }
};

TEST(AgradRev,asinh_NaN) {
  asinh_fun asinh_;
  test_nan(asinh_,false,true);
}
