#include <stan/diff/rev/operator_unary_increment.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,plus_plus_a) {
  AVAR a(5.0);
  EXPECT_FLOAT_EQ(5.0,a.val());
  AVAR f = ++a;
  EXPECT_FLOAT_EQ(6.0,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(DiffRev,plus_plus_a_2) {
  AVAR a(5.0);
  EXPECT_FLOAT_EQ(5.0,a.val());
  AVAR f = ++a;
  EXPECT_FLOAT_EQ(6.0,f.val());
  EXPECT_FLOAT_EQ(6.0,a.val());

  // see next test when created later
  AVEC x = createAVEC(a); 

  ++a;
  EXPECT_FLOAT_EQ(7.0,a.val());
  EXPECT_FLOAT_EQ(6.0,f.val());

  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(DiffRev,plus_plus_a_3) {
  AVAR a(5.0);
  AVAR f = ++a;
  ++a; // reassignment loses connection to f
  AVEC x = createAVEC(a); 
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0,g[0]);
}

TEST(DiffRev,a_plus_plus) {
  AVAR a(5.0);
  AVEC x = createAVEC(a); // compare to placement in test 2
  AVAR f = a++;
  EXPECT_FLOAT_EQ(6.0,a.val());
  EXPECT_FLOAT_EQ(5.0,f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(DiffRev,a_plus_plus_2) {
  AVAR a(5.0);
  AVAR f = a++;
  AVEC x = createAVEC(a); // compare to placement in test 1
  EXPECT_FLOAT_EQ(6.0,a.val());
  EXPECT_FLOAT_EQ(5.0,f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0,g[0]);
}
