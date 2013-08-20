#include <stan/agrad/rev/asin.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,asin_var) {
  AVAR a = 0.68;
  AVAR f = asin(a);
  EXPECT_FLOAT_EQ(asin(0.68), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/sqrt(1.0 - (0.68 * 0.68)), g[0]);
}

TEST(AgradRev,asin_1) {
  AVAR a = 1;
  AVAR f = asin(a);
  EXPECT_FLOAT_EQ((1.57079632679),f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/sqrt(1.0 - (1*1)),g[0]);
}

TEST(AgradRev,asin_neg_1) {
  AVAR a = -1;
  AVAR f = asin(a);
  EXPECT_FLOAT_EQ((-1.57079632679),f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/sqrt(1.0 - (-1*-1)),g[0]);
}

TEST(AgradRev,asin_out_of_bounds) {
  AVAR a = 2;
  EXPECT_THROW(asin(a),std::domain_error)
    <<"asin(2) should throw error";

  a = -2;
  EXPECT_THROW(asin(a),std::domain_error)
    <<"asin(-2) should throw error";
}
