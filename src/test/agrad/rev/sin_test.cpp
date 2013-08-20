#include <stan/agrad/rev/sin.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,sin_var) {
  AVAR a = 0.49;
  AVAR f = sin(a);
  EXPECT_FLOAT_EQ((.470625888), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(cos(0.49),g[0]);
}

TEST(AgradRev,sin_neg_var) {
  AVAR a = -0.49;
  AVAR f = sin(a);
  EXPECT_FLOAT_EQ((-.470625888), f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(cos(-0.49), g[0]);
}

TEST(AgradRev,sin_boundry) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  EXPECT_THROW(sin(a),std::domain_error)
    <<"sin(a) should throw error";

  AVAR b = -inf;
  EXPECT_THROW(sin(b),std::domain_error)
    <<"sin(b) should throw error";
}
