#include <stan/agrad/rev/functions/gamma_q.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <test/unit/agrad/util.hpp>
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
TEST(AgradRevGammaQ,infLoopInVersion2_0_1_var_var) {
  // FIXME: causes infinite loop in 2.0.1 gradient calcs
  AVAR a = 8.01006;
  AVAR b = 2.47579e+215;
  AVEC x = createAVEC(a,b);

  AVAR f = gamma_q(a,b);
  VEC g;
  EXPECT_THROW(f.grad(x,g), std::domain_error);
}
TEST(AgradRevGammaQ,infLoopInVersion2_0_1_var_double) {
  // FIXME: causes infinite loop in 2.0.1 gradient calcs
  AVAR a = 8.01006;
  double b = 2.47579e+215;
  AVEC x = createAVEC(a);

  AVAR f = gamma_q(a,b);
  VEC g;
  EXPECT_THROW(f.grad(x,g), std::domain_error);
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

TEST(AgradRev,gamma_q_nan_vv) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR b = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::gamma_q(a,b);

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(2U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
  EXPECT_TRUE(boost::math::isnan(g[1]));
}

TEST(AgradRev,gamma_q_nan_vd) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::gamma_q(a,1);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}

TEST(AgradRev,gamma_q_nan_dv) {
  AVAR a = std::numeric_limits<double>::quiet_NaN();
  AVAR f = stan::agrad::gamma_q(1,a);

  AVEC x = createAVEC(a);
  VEC g;
  f.grad(x,g);
  
  EXPECT_TRUE(boost::math::isnan(f.val()));
  ASSERT_EQ(1U,g.size());
  EXPECT_TRUE(boost::math::isnan(g[0]));
}
