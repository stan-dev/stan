#include <stan/agrad/rev/tan.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/constants.hpp>

TEST(AgradRev,tan_var) {
  AVAR a = 0.68;
  AVAR f = tan(a);
  EXPECT_FLOAT_EQ(0.80866137, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1 + tan(0.68)*tan(0.68), g[0]);
}

TEST(AgradRev,tan_neg_var) {
  AVAR a = -.68;
  AVAR f = tan(a);
  EXPECT_FLOAT_EQ(-0.80866137, f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1 + tan(-0.68)*tan(-0.68), g[0]);
}

TEST(AgradRev,tan_boundry) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  EXPECT_THROW(tan(a),std::domain_error)
    <<"tan(a) should throw error";

  AVAR b = -inf;
  EXPECT_THROW(tan(b),std::domain_error)
    <<"tan(b) should throw error";
}

