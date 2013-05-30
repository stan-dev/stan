#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>

TEST(AgradFvar, log_falling_factorial) {
  using stan::agrad::fvar;
  using stan::agrad::log_falling_factorial;
  using boost::math::digamma;

  fvar<double> a(4.0,1.0);
  fvar<double> x = log_falling_factorial(a,1);
  EXPECT_FLOAT_EQ(std::log(24.0), x.val_);
  EXPECT_FLOAT_EQ(digamma(5), x.d_);

  fvar<double> c(-3.0,2.0);

  EXPECT_THROW(log_falling_factorial(c, 2), std::domain_error);
  EXPECT_THROW(log_falling_factorial(2, c), std::domain_error);
  EXPECT_THROW(log_falling_factorial(c, c), std::domain_error);

  x = log_falling_factorial(a,a);
  EXPECT_FLOAT_EQ(0.0, x.val_);
  EXPECT_FLOAT_EQ(0.0, x.d_);

  x = log_falling_factorial(5, a);
  EXPECT_FLOAT_EQ(std::log(5.0), x.val_);
  EXPECT_FLOAT_EQ(-digamma(5.0),x.d_);
}
