#include <stan/agrad/rev/functions/log_rising_factorial.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>

TEST(AgradRev,log_rising_factorial_var_double) {
  double a(1);
  AVAR b(4.0);
  AVAR f = stan::agrad::log_rising_factorial(b,a);
  EXPECT_FLOAT_EQ(std::log(4.0),f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(boost::math::digamma(5) - boost::math::digamma(4),g[1]);
}

TEST(AgradRev, log_rising_factorial_exceptions) {
  double a(1);
  AVAR b(-3.0);
  EXPECT_THROW(stan::agrad::log_rising_factorial(b,a), std::domain_error);
  EXPECT_THROW(stan::agrad::log_rising_factorial(a,b), std::domain_error);
  EXPECT_THROW(stan::agrad::log_rising_factorial(b,b), std::domain_error);
}

TEST(AgradRev, log_rising_factorial_double_var) {
  double a(5.0);
  AVAR b(4.0);
  AVAR f = stan::agrad::log_rising_factorial(a,b);
  EXPECT_FLOAT_EQ(std::log(5*6*7*8), f.val());
  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(boost::math::digamma(9), g[1]);
}

TEST(AgradRev, log_rising_factorial_var_var) {
  AVAR c(5.0);
  AVAR b(4.0);
  AVAR f = stan::agrad::log_rising_factorial(b,c);
  EXPECT_FLOAT_EQ(std::log(4*5*6*7*8), f.val());
  AVEC x = createAVEC(b,c);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(boost::math::digamma(9.0) - boost::math::digamma(4.0), g[0]);
  EXPECT_FLOAT_EQ(boost::math::digamma(9), g[1]);
}
