#include <stan/agrad/rev/gamma_q.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,gamma_q_var_var) {
  AVAR a = 0.5;
  AVAR b = 1.0;
  AVAR f = gamma_q(a,b);
  EXPECT_FLOAT_EQ(boost::math::gamma_q(0.5,1.0),f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.38983709, g[0]);
  EXPECT_FLOAT_EQ(-boost::math::gamma_p_derivative(0.5,1.0), g[1]);
  
  a = -0.5;
  EXPECT_THROW(gamma_q(a,b), std::domain_error);

  b = -1.0;
  EXPECT_THROW(gamma_q(a,b), std::domain_error);
}
TEST(AgradRev,gamma_q_double_var) {
  double a = 0.5;
  AVAR b = 1.0;
  AVAR f = gamma_q(a,b);
  EXPECT_FLOAT_EQ(boost::math::gamma_q(0.5,1.0),f.val());

  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-boost::math::gamma_p_derivative(0.5,1.0), g[0]);

  a = -0.5;
  EXPECT_THROW(gamma_q(a,b), std::domain_error);

  b = -1.0;
  EXPECT_THROW(gamma_q(a,b), std::domain_error);
}
TEST(AgradRev,gamma_q_var_double) {
  AVAR a = 0.5;
  double b = 1.0;
  AVAR f = gamma_q(a,b);
  EXPECT_FLOAT_EQ(boost::math::gamma_q(0.5,1.0),f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0.38983709, g[0]);

  a = -0.5;
  EXPECT_THROW(gamma_q(a,b), std::domain_error);

  b = -1.0;
  EXPECT_THROW(gamma_q(a,b), std::domain_error);
}
