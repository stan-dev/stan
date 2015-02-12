#include <stan/math/rev/scal/fun/exp.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev,exp_a) {
  AVAR a(6.0);
  AVAR f = exp(a); // mix exp() functs w/o namespace
  EXPECT_FLOAT_EQ(exp(6.0),f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(exp(6.0),g[0]);
}

struct exp_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return exp(arg1);
  }
};

TEST(AgradRev,exp_NaN) {
  exp_fun exp_;
  test_nan(exp_,false,true);
}
