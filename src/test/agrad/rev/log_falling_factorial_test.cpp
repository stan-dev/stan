#include <stan/agrad/rev/log_falling_factorial.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>

TEST(AgradRev,log_falling_factorial) {
  double a(1);
  AVAR b(4.0);
  AVAR f = stan::agrad::log_falling_factorial(b,a);
  EXPECT_FLOAT_EQ(std::log(24.0),f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(boost::math::digamma(5),g[1]);

  a = 1;
  b = -3.0;
  EXPECT_THROW(stan::agrad::log_falling_factorial(b,a), std::domain_error);
  EXPECT_THROW(stan::agrad::log_falling_factorial(a,b), std::domain_error);
  EXPECT_THROW(stan::agrad::log_falling_factorial(b,b), std::domain_error);

  a = 5;
  b = 4.0;
  f = stan::agrad::log_falling_factorial(a,b);
  EXPECT_FLOAT_EQ(std::log(5.0), f.val());
  x = createAVEC(a,b);
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(-boost::math::digamma(5), g[1]);

  f = stan::agrad::log_falling_factorial(b,b);
  EXPECT_FLOAT_EQ(0.0, f.val());
  x = createAVEC(b,b);
  f.grad(x,g);
  EXPECT_FLOAT_EQ(-boost::math::digamma(5), g[0]);
  EXPECT_FLOAT_EQ(-boost::math::digamma(5), g[1]);
}
