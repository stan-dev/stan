#include <stan/agrad/rev/var_stack.hpp>
#include <stan/agrad/rev/operators/operator_multiplication.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>

// just test basic autodiff;  no more free_memory operation
TEST(AgradRev,varStack) { 
  AVAR a = 2.0;
  AVAR b = -3.0;
  AVAR f = a * b;
  EXPECT_FLOAT_EQ(-6.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-3.0,grad_f[0]);
  EXPECT_FLOAT_EQ(2.0,grad_f[1]);

  AVAR aa = 2.0;
  AVAR bb = -3.0;
  AVAR ff = aa * bb;
  EXPECT_FLOAT_EQ(-6.0,ff.val());

  AVEC xx = createAVEC(aa,bb);
  VEC grad_ff;
  ff.grad(xx,grad_ff);
  EXPECT_FLOAT_EQ(-3.0,grad_ff[0]);
  EXPECT_FLOAT_EQ(2.0,grad_ff[1]);
}

TEST(AgradRev, recoverMemoryLogicError) {
  stan::agrad::start_nested();
  EXPECT_THROW(stan::agrad::recover_memory(), std::logic_error);
  stan::agrad::recover_memory_nested(); // clean up for next test
}

TEST(AgradRev, recoverMemoryNestedLogicError) {
  EXPECT_THROW(stan::agrad::recover_memory_nested(), std::logic_error);
}
