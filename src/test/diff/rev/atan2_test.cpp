#include <stan/diff/rev/atan2.hpp>
#include <stan/diff/rev/operator_division.hpp>
#include <stan/diff/rev/operator_multiplication.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,atan2_var_var) {
  AVAR a = 1.2;
  AVAR b = 3.9;
  AVAR f = atan2(a,b);
  EXPECT_FLOAT_EQ(atan2(1.2,3.9),f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(3.9 / (1.2 * 1.2 + 3.9 * 3.9), g[0]);
  EXPECT_FLOAT_EQ(-1.2 / (1.2 * 1.2 + 3.9 * 3.9), g[1]);
}

TEST(DiffRev,atan2_dvd) {
  AVAR sigma = 1;
  AVEC x = createAVEC(sigma);
  AVAR f = atan2(1.0,sigma) / 3.14;
  VEC g;
  f.grad(x,g);

  AVAR sigma1 = 1;
  AVEC x1 = createAVEC(sigma1);
  AVAR f1 = atan2(1.0,sigma1);
  VEC g1;
  f1.grad(x1,g1);

  EXPECT_FLOAT_EQ(3.14 * g[0],g1[0]);
}
TEST(DiffRev,atan2_var_var__integration) {
  double c = 5.0;
  AVAR a = 1.2;
  AVAR b = 3.9;
  AVAR f = atan2(a,b) * c;
  EXPECT_FLOAT_EQ(atan2(1.2,3.9)*c,f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(3.9 / (1.2 * 1.2 + 3.9 * 3.9) * c, g[0]);
  EXPECT_FLOAT_EQ(-1.2 / (1.2 * 1.2 + 3.9 * 3.9) * c, g[1]);
}


TEST(DiffRev,atan2_var_double) {
  AVAR a = 1.2;

  double b = 3.9;
  AVAR f = atan2(a,b);
  EXPECT_FLOAT_EQ(atan2(1.2,3.9),f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(3.9 / (1.2 * 1.2 + 3.9 * 3.9), g[0]);
}

TEST(DiffRev,atan2_double_var) {
  double a = 1.2;
  AVAR b = 3.9;
  AVAR f = atan2(a,b);
  EXPECT_FLOAT_EQ(atan2(1.2,3.9),f.val());

  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-1.2 / (1.2 * 1.2 + 3.9 * 3.9), g[0]);
}
