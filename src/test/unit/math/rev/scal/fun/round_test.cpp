#include <stan/math/rev/scal/fun/round.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>

TEST(AgradRev,round) {
  AVAR a = 1.2;
  AVAR f = round(a);
  EXPECT_FLOAT_EQ(1.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0, grad_f[0]);
}

TEST(AgradRev,round_2) {
  AVAR a = -1.2;
  AVAR f = round(a);
  EXPECT_FLOAT_EQ(-1.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0, grad_f[0]);
}

TEST(AgradRev,round_3) {
  AVAR a = 1.7;
  AVAR f = round(a);
  EXPECT_FLOAT_EQ(2.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0, grad_f[0]);
}


TEST(AgradRev,round_4) {
  AVAR a = -1.7;
  AVAR f = round(a);
  EXPECT_FLOAT_EQ(-2.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0, grad_f[0]);
}

struct round_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return round(arg1);
  }
};

TEST(AgradRev,round_NaN) {
  round_fun round_;
  test_nan(round_,false,true);
}
