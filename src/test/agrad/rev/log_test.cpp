#include <stan/agrad/rev/log.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,log_a) {
  AVAR a(5.0);
  AVAR f = log(a); 
  EXPECT_FLOAT_EQ(log(5.0),f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0/5.0,g[0]);
}

TEST(AgradRev,log_inf) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR a = inf;
  EXPECT_THROW(log(a),std::domain_error);
}

TEST(AgradRev,log_inf_2) {
  double inf = std::numeric_limits<double>::infinity();
  AVAR f = log(inf);
  AVEC x = createAVEC(inf);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.0,g[0]);
}

TEST(AgradRev,log_0) {
  AVAR a(0.0);
  EXPECT_THROW(log(a),std::domain_error);
}

TEST(AgradRev,log_neg){
  AVAR a(-2.0);
  EXPECT_THROW(log(a),std::domain_error);
}
