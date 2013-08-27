#include <stan/diff/rev/log2.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>
#include <valarray>
#include <stan/diff/rev/numeric_limits.hpp>

TEST(DiffRev,log2) {
  AVAR a = 3.0;
  AVAR f = stan::diff::log2(a);
  EXPECT_FLOAT_EQ(std::log(3.0)/std::log(2.0), f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0 / 3.0 / std::log(2.0), grad_f[0]);

  a = std::numeric_limits<AVAR>::infinity();
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),
                  stan::diff::log2(a).val());
}
