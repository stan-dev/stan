#include <stan/agrad/rev/functions/tgamma.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <test/unit/agrad/rev/nan_util.hpp>

TEST(AgradRev,tgamma) {
  AVAR a = 3.5;
  AVAR f = tgamma(a);
  EXPECT_FLOAT_EQ(boost::math::tgamma(3.5),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(boost::math::digamma(3.5) * boost::math::tgamma(3.5),grad_f[0]);
}  

struct tgamma_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return tgamma(arg1);
  }
};

TEST(AgradRev,tgamma_NaN) {
  tgamma_fun tgamma_;
  test_nan(tgamma_,false,true);
}
