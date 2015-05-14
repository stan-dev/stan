#include <stan/math/rev/scal/fun/log_rising_factorial.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>

TEST(AgradRev,log_rising_factorial_var_double) {
  double a(1);
  AVAR b(4.0);
  AVAR f = stan::math::log_rising_factorial(b,a);
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
  EXPECT_THROW(stan::math::log_rising_factorial(b,a), std::domain_error);
  EXPECT_THROW(stan::math::log_rising_factorial(a,b), std::domain_error);
  EXPECT_THROW(stan::math::log_rising_factorial(b,b), std::domain_error);
}

TEST(AgradRev, log_rising_factorial_double_var) {
  double a(5.0);
  AVAR b(4.0);
  AVAR f = stan::math::log_rising_factorial(a,b);
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
  AVAR f = stan::math::log_rising_factorial(b,c);
  EXPECT_FLOAT_EQ(std::log(4*5*6*7*8), f.val());
  AVEC x = createAVEC(b,c);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(boost::math::digamma(9.0) - boost::math::digamma(4.0), g[0]);
  EXPECT_FLOAT_EQ(boost::math::digamma(9), g[1]);
}

struct log_rising_factorial_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return log_rising_factorial(arg1,arg2);
  }
};

TEST(AgradRev, log_rising_factorial_nan) {
  log_rising_factorial_fun log_rising_factorial_;
  test_nan(log_rising_factorial_,3.0,5.0,false,true);
}
