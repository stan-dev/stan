#include <stan/math/rev/scal/fun/step.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,step) {
  AVAR a = 3.5;
  AVAR f = stan::math::step(a);
  EXPECT_FLOAT_EQ(1.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  

TEST(AgradRev,step_2) {
  AVAR a = 0.0;
  AVAR f = stan::math::step(a);
  EXPECT_FLOAT_EQ(1.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  

TEST(AgradRev,step_3) {
  AVAR a = -18765.3;
  AVAR f = stan::math::step(a);
  EXPECT_FLOAT_EQ(0.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  
TEST(AgradRev,step_nan) {
  stan::math::var nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_EQ(1U, stan::math::step(nan).val());
}
