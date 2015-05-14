#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>

struct foo : public stan::math::chainable_alloc {
  std::vector<double> x_;
  foo() : x_(3) { }
  ~foo() { }
  void chain() { }
};

TEST(AgradRev, varStackRecoverNestedSegFaultFix) {
  // this test failed in 2.5, but passes in 2.6
  stan::math::start_nested();
  foo* x = new foo();
  x->chain();
  stan::math::recover_memory_nested();
  // should be able to do this redundantly:
  stan::math::recover_memory(); 
}

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
  stan::math::start_nested();
  EXPECT_THROW(stan::math::recover_memory(), std::logic_error);
  stan::math::recover_memory_nested(); // clean up for next test
}

TEST(AgradRev, recoverMemoryNestedLogicError) {
  EXPECT_THROW(stan::math::recover_memory_nested(), std::logic_error);
}
