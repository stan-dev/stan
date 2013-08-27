#include <stan/diff/rev/var_stack.hpp>
#include <stan/diff/rev/operator_multiplication.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,free_memory) {
  AVAR a = 2.0;
  AVAR b = -3.0;
  AVAR f = a * b;
  EXPECT_FLOAT_EQ(-6.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-3.0,grad_f[0]);
  EXPECT_FLOAT_EQ(2.0,grad_f[1]);
  stan::diff::free_memory();

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
