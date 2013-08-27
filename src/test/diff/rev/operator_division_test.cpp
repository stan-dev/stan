#include <stan/diff/rev/operator_division.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,a_div_b) {
  AVAR a = 6.0;
  AVAR b = 3.0;
  AVAR f = a / b;
  EXPECT_FLOAT_EQ(2.0,f.val());
  
  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/3.0,g[0]);
  EXPECT_FLOAT_EQ(-6.0/(3.0*3.0),g[1]);
}

TEST(DiffRev,a_divide_bd) {
  AVAR a = 6.0;
  double b = 3.0;
  AVAR f = a / b;
  EXPECT_FLOAT_EQ(2.0,f.val());
  
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/3.0,g[0]);
}

TEST(DiffRev,ad_divide_b) {
  double a = 6.0;
  AVAR b = 3.0;
  AVAR f = a / b;
  EXPECT_FLOAT_EQ(2.0,f.val());
  
  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-6.0/(3.0*3.0),g[0]);
}
