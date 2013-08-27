#include <stan/diff/rev/operator_unary_plus.hpp>
#include <stan/diff/rev/operator_equal.hpp>
#include <test/diff/util.hpp>

#include <gtest/gtest.h>

TEST(DiffRev,pos_a) {
  AVAR a = 5.0;
  AVAR f = +a;
  EXPECT_FLOAT_EQ(5.0,f.val());
  EXPECT_TRUE(a == +a);
  AVEC x = createAVEC(a);
  VEC dx;
  f.grad(x,dx);
  EXPECT_FLOAT_EQ(1.0,dx[0]);
}
