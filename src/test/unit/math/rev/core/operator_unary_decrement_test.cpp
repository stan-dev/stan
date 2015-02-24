#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev,minus_minus_a) {
  AVAR a(5.0);
  AVAR f = --a;
  EXPECT_FLOAT_EQ(4.0,f.val());
  EXPECT_FLOAT_EQ(4.0,a.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradRev,minus_minus_a_2) {
  AVAR a(5.0);
  AVEC x = createAVEC(a);
  AVAR f = --a;
  EXPECT_FLOAT_EQ(4.0,f.val());
  EXPECT_FLOAT_EQ(4.0,a.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradRev,a_minus_minus) {
  AVAR a(5.0);
  AVEC x = createAVEC(a); // compare to placement in test 2
  AVAR f = a--;
  EXPECT_FLOAT_EQ(4.0,a.val());
  EXPECT_FLOAT_EQ(5.0,f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(AgradRev,a_minus_minus_2) {
  AVAR a(5.0);
  AVAR f = a--;
  AVEC x = createAVEC(a); // compare to placement in test 1
  EXPECT_FLOAT_EQ(4.0,a.val());
  EXPECT_FLOAT_EQ(5.0,f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0,g[0]);
}


struct minus_minus_fun {
  template <typename T0>
  inline T0
  operator()(T0 arg1) const {
    return (arg1--);
  }
};

TEST(AgradRev, minus_minus_nan) {
  minus_minus_fun minus_minus_;

  test_nan(minus_minus_,false, true);
}
