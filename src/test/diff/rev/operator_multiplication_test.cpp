#include <stan/diff/rev/operator_multiplication.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,a_times_b) {
  AVAR a = 2.0;
  AVAR b = -3.0;
  AVAR f = a * b;
  EXPECT_FLOAT_EQ(-6.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-3.0,grad_f[0]);
  EXPECT_FLOAT_EQ(2.0,grad_f[1]);
}

TEST(DiffRev,a_times_a) {
  AVAR a = 2.0;
  AVAR f = a * a;
  EXPECT_FLOAT_EQ(4.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(4.0,grad_f[0]);
}

TEST(DiffRev,a_times_y) {
  AVAR a = 2.0;
  double y = -3.0;
  AVAR f = a * y;
  EXPECT_FLOAT_EQ(-6.0,f.val());
  
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-3.0,g[0]);
}
 
TEST(DiffRev,x_times_b) {
  double x = 2.0;
  AVAR b = -3.0;
  AVAR f = x * b;
  EXPECT_FLOAT_EQ(-6.0,f.val());

  AVEC v = createAVEC(b);
  VEC g;
  f.grad(v,g);
  EXPECT_FLOAT_EQ(2.0,g[0]);
}
