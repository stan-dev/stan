#include <stan/agrad/rev/functions/gamma_p.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,gamma_p_var_var) {
  AVAR a = 0.5;
  AVAR b = 1.0;
  AVAR f = gamma_p(a,b);
  EXPECT_FLOAT_EQ(boost::math::gamma_p(0.5,1.0),f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-0.389837, g[0]);
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), g[1]);
  
  a = -0.5;
  EXPECT_THROW(gamma_p(a,b), std::domain_error);

  b = -1.0;
  EXPECT_THROW(gamma_p(a,b), std::domain_error);
}
TEST(AgradRev,gamma_p_double_var) {
  double a = 0.5;
  AVAR b = 1.0;
  AVAR f = gamma_p(a,b);
  EXPECT_FLOAT_EQ(boost::math::gamma_p(0.5,1.0),f.val());

  AVEC x = createAVEC(b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), g[0]);

  a = -0.5;
  EXPECT_THROW(gamma_p(a,b), std::domain_error);

  b = -1.0;
  EXPECT_THROW(gamma_p(a,b), std::domain_error);
}
TEST(AgradRev,gamma_p_var_double) {
  AVAR a = 0.5;
  double b = 1.0;
  AVAR f = gamma_p(a,b);
  EXPECT_FLOAT_EQ(boost::math::gamma_p(0.5,1.0),f.val());

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-0.389837, g[0]);

  a = -0.5;
  EXPECT_THROW(gamma_p(a,b), std::domain_error);

  b = -1.0;
  EXPECT_THROW(gamma_p(a,b), std::domain_error);
}
