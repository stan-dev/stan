#include <stan/math/rev/scal/fun/cbrt.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev,cbrt) {
  AVAR a = 27.0;
  AVAR f = cbrt(a);
  EXPECT_FLOAT_EQ(3.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0 / 3.0 / std::pow(27.0,2.0/3.0), grad_f[0]);
}

struct cbrt_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return cbrt(arg1);
  }
};

TEST(AgradRev,cbrt_NaN) {
  cbrt_fun cbrt_;
  test_nan(cbrt_,false,true);
}
