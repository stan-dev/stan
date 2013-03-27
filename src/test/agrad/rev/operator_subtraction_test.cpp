#include <stan/agrad/rev/operator_subtraction.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,a_minus_b) {
  AVAR a = 5.0;
  AVAR b = 2.0;
  AVAR f = a - b;
  EXPECT_FLOAT_EQ(3.0,f.val());
  AVEC x = createAVEC(a,b);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
  EXPECT_FLOAT_EQ(-1.0,dx[1]);
}

TEST(AgradRev,a_minus_a) {
  AVAR a = 5.0;
  AVAR f = a - a;
  EXPECT_FLOAT_EQ(0.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(0.0,dx[0]);
}

TEST(AgradRev,a_minus_x) {
  AVAR a = 5.0;
  double z = 3.0;
  AVAR f = a - z;
  EXPECT_FLOAT_EQ(2.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
}

TEST(AgradRev,x_minus_a) {
  AVAR a = 2.0;
  double z = 5.0;
  AVAR f = z - a;
  EXPECT_FLOAT_EQ(3.0,f.val());
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(-1.0,dx[0]);
}
