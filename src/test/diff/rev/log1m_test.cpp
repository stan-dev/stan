#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/diff/rev/log1m.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,log1m) {
  AVAR a = 0.1;
  AVAR f = log1m(a);
  EXPECT_FLOAT_EQ(log(1 - 0.1), f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-1.0/(1.0 - 0.1), grad_f[0]);
}
TEST(DiffRev,log1mErr) {
  AVAR a = 10;
  AVAR f = log1m(a);
  EXPECT_TRUE(boost::math::isnan(f.val()));
}
