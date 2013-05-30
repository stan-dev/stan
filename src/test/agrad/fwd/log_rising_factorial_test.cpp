#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>

TEST(AgradFvar, log_rising_factorial) {
  using stan::agrad::fvar;
  using stan::agrad::log_rising_factorial;
  using boost::math::digamma;

  fvar<double> a(4.0,1.0);
  fvar<double> x = log_rising_factorial(a,1.0);
  EXPECT_FLOAT_EQ(std::log(4.0), x.val_);
  EXPECT_FLOAT_EQ(0.25, x.d_);

  fvar<double> c(-3.0,2.0);

  EXPECT_THROW(log_rising_factorial(c, 2), std::domain_error);
  EXPECT_THROW(log_rising_factorial(2, c), std::domain_error);
  EXPECT_THROW(log_rising_factorial(c, c), std::domain_error);

  x = log_rising_factorial(a,a);
  EXPECT_FLOAT_EQ(std::log(840.0), x.val_);
  EXPECT_FLOAT_EQ((2 * digamma(8) - digamma(4)), x.d_);

  x = log_rising_factorial(5, a);
  EXPECT_FLOAT_EQ(std::log(1680.0), x.val_);
  EXPECT_FLOAT_EQ(digamma(9), x.d_);
}
