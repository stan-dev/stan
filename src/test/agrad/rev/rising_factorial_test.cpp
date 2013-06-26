#include <stan/agrad/rev/rising_factorial.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>

TEST(AgradRev,rising_factorial_var_double) {
  using boost::math::digamma;
  double a(1);
  AVAR b(4.0);
  AVAR f = stan::agrad::rising_factorial(b,a);
  EXPECT_FLOAT_EQ(4,f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ((digamma(5.0) - digamma(4.0)) * 4.0,g[1]);
} 

TEST(AgradRev, rising_factorial_exceptions) {
  double a(1);
  AVAR b(-3.0);
  EXPECT_THROW(stan::agrad::rising_factorial(b,a), std::domain_error);
  EXPECT_THROW(stan::agrad::rising_factorial(a,b), std::domain_error);
  EXPECT_THROW(stan::agrad::rising_factorial(b,b), std::domain_error);
}

TEST(AgradRev, rising_factorial_double_var) {
  using boost::math::digamma;
  double a(5);
  AVAR b(4.0);
  AVAR f = stan::agrad::rising_factorial(a,b);
  EXPECT_FLOAT_EQ(5*6*7*8, f.val());
  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(digamma(9) * 5*6*7*8, g[1]);
}

TEST(AgradRev, rising_factorial_var_var) {
  using boost::math::digamma;
  AVAR c(4.0);
  AVAR b(4.0);
  AVAR f = stan::agrad::rising_factorial(b,c);
  EXPECT_FLOAT_EQ(4*5*6*7, f.val());
  AVEC x = createAVEC(b,c);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(4.0*5.0*6.0*7.0 * (digamma(8.0) - digamma(4.0)), g[0]);
  EXPECT_FLOAT_EQ(4.0*5.0*6.0*7.0 * digamma(8), g[1]);
}
