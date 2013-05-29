#include <stan/agrad/rev/rising_factorial.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>

TEST(AgradRev,rising_factorial_var_double) {
  double a(1);
  AVAR b(4.0);
  AVAR f = stan::agrad::rising_factorial(b,a);
  EXPECT_FLOAT_EQ(4,f.val());

  AVEC x = createAVEC(a,b);
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(1,g[1]);


  a = 1;
  b = -3.0;
  EXPECT_THROW(stan::agrad::rising_factorial(b,a), std::domain_error);
}
