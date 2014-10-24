#include <stan/agrad/rev/functions/log_falling_factorial.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <test/unit/agrad/rev/nan_util.hpp>
#include <stan/meta/traits.hpp>


TEST(AgradRev,log_falling_factorial_var_double) {
  double a(1);
  AVAR b(4.0);
  AVAR f = stan::agrad::log_falling_factorial(b,a);
  EXPECT_FLOAT_EQ(std::log(24.0),f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(boost::math::digamma(5),g[1]);
}

TEST(AgradRev, log_falling_factorial_exceptions) {
  double a(1);
  AVAR b(-3.0);
  EXPECT_THROW(stan::agrad::log_falling_factorial(b,a), std::domain_error);
  EXPECT_THROW(stan::agrad::log_falling_factorial(a,b), std::domain_error);
  EXPECT_THROW(stan::agrad::log_falling_factorial(b,b), std::domain_error);
}

TEST(AgradRev, log_falling_factorial_double_var) {
  double a(5);
  AVAR b(4.0);
  AVAR f = stan::agrad::log_falling_factorial(a,b);
  EXPECT_FLOAT_EQ(std::log(5.0), f.val());
  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(-boost::math::digamma(5), g[1]);
}

TEST(AgradRev, log_falling_factorial_var_var) {
  AVAR b(4.0);
  AVAR c(4.0);
  AVAR f = stan::agrad::log_falling_factorial(b,c);
  EXPECT_FLOAT_EQ(0.0, f.val());
  AVEC x = createAVEC(b,c);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(boost::math::digamma(5), g[0]);
  EXPECT_FLOAT_EQ(-boost::math::digamma(5), g[1]);
}

struct log_falling_factorial_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return log_falling_factorial(arg1,arg2);
  }
};

TEST(AgradRev, log_falling_factorial_nan) {
  log_falling_factorial_fun log_falling_factorial_;
  test_nan(log_falling_factorial_,4.0,4.0,false,true);
}
