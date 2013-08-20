#include <stan/agrad/rev/sqrt.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,sqrt_a) {
  AVAR a(5.0);
  AVAR f = sqrt(a); 
  EXPECT_FLOAT_EQ(sqrt(5.0),f.val());
  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ((1.0/2.0) * pow(5.0,-0.5), g[0]);
}

TEST(AgradRev,sqrt_neg) {
  AVAR a = (-1.0);
  EXPECT_THROW(sqrt(a),std::domain_error)
    <<"sqrt(-1) should throw error";

  a = -100;
  EXPECT_THROW(sqrt(a),std::domain_error)
    <<"sqrt(-100) should throw error";
}

TEST(AgradRev,sqrt_inf) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  AVAR f =sqrt(a);
  EXPECT_FLOAT_EQ(inf,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0,g[0]);
}

TEST(AgradRev,sqrt_zero) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a(0.0);
  AVAR f = sqrt(a);
  EXPECT_FLOAT_EQ(0.0,f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(inf,g[0]);
}
