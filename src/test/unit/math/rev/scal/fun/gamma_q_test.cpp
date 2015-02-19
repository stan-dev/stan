#include <stan/math/rev/scal/fun/gamma_q.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>

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

struct gamma_q_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return gamma_q(arg1,arg2);
  }
};

TEST(AgradRev, gamma_q_nan) {
  gamma_q_fun gamma_q_;
  test_nan(gamma_q_,3.0,5.0,false,true);
}
