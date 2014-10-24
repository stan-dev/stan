#include <stan/agrad/rev/functions/lgamma.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <test/unit/agrad/rev/nan_util.hpp>

TEST(AgradRev,lgamma) {
  AVAR a = 3.0;
  AVAR f = lgamma(a);
  EXPECT_FLOAT_EQ(lgamma(3.0),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(boost::math::digamma(3.0),grad_f[0]);
}

struct lgamma_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return lgamma(arg1);
  }
};

TEST(AgradRev,lgamma_NaN) {
  lgamma_fun lgamma_;
  test_nan(lgamma_,false,true);
}
