#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/agrad/rev/functions/log1m.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/rev/nan_util.hpp>

TEST(AgradRev,log1m) {
  AVAR a = 0.1;
  AVAR f = log1m(a);
  EXPECT_FLOAT_EQ(log(1 - 0.1), f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-1.0/(1.0 - 0.1), grad_f[0]);
}
TEST(AgradRev,log1mErr) {
  AVAR a = 10;
  AVAR f = log1m(a);
  EXPECT_TRUE(boost::math::isnan(f.val()));
}

struct log1m_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log1m(arg1);
  }
};

TEST(AgradRev,log1m_NaN) {
  log1m_fun log1m_;
  test_nan(log1m_,false,true);
}
