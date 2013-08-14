#include <stan/diff/rev/exp2.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>
#include <stan/diff/rev/numeric_limits.hpp>

TEST(DiffRev,exp2) {
  AVAR a = 1.3;
  AVAR f = stan::diff::exp2(a);
  EXPECT_FLOAT_EQ(std::pow(2.0,1.3), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::pow(2.0,1.3) * std::log(2.0),grad_f[0]);
  
  a = std::numeric_limits<AVAR>::infinity();
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),
                  stan::diff::exp2(a).val());
}
