#include <stan/math/rev/scal/fun/falling_factorial.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <test/unit/math/rev/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>

TEST(AgradRev,falling_factorial_var_double) {
  double a(1);
  AVAR b(4.0);
  AVAR f = stan::math::falling_factorial(b,a);
  EXPECT_FLOAT_EQ(24,f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(boost::math::digamma(5) * 24.0,g[1]);
}

TEST(AgradRev, falling_factorial_exceptions) {
  double a = 1;
  AVAR b(-3.0);
  EXPECT_THROW(stan::math::falling_factorial(b,a), std::domain_error);
  EXPECT_THROW(stan::math::falling_factorial(a,b), std::domain_error);
  EXPECT_THROW(stan::math::falling_factorial(b,b), std::domain_error);
}

TEST(AgradRev, falling_factorial_double_var) {
  double a(5);
  AVAR b(4.0);
  AVAR f = stan::math::falling_factorial(a,b);
  EXPECT_FLOAT_EQ(5.0, f.val());
  AVEC x = createAVEC(a,b);
  VEC g;  
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(boost::math::digamma(5) * -5.0, g[1]);
}
TEST(AgradRev, falling_factorial_var_var) {
  AVAR a(4.0);
  AVAR b(4.0);
  AVAR f = stan::math::falling_factorial(a,b);
  EXPECT_FLOAT_EQ(1.0, f.val());
  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(1.0 * boost::math::digamma(5), g[0]);
  EXPECT_FLOAT_EQ(boost::math::digamma(5) * -1.0, g[1]);
}

struct falling_factorial_fun {
  template <typename T0, typename T1>
  inline 
  typename stan::return_type<T0,T1>::type
  operator()(const T0& arg1,
             const T1& arg2) const {
    return falling_factorial(arg1,arg2);
  }
};

TEST(AgradRev, falling_factorial_nan) {
  falling_factorial_fun falling_factorial_;
  test_nan(falling_factorial_,4.0,1.0,false,true);

}
