#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>

TEST(AgradFvar, rising_factorial) {
  using stan::agrad::fvar;
  using stan::agrad::rising_factorial;
  using boost::math::digamma;

  fvar<double> a(4.0,1.0);
  fvar<double> x = rising_factorial(a,1);
  EXPECT_FLOAT_EQ(4.0, x.val_);
  EXPECT_FLOAT_EQ(1.0, x.d_);

  fvar<double> c(-3.0,2.0);

  EXPECT_THROW(rising_factorial(c, 2), std::domain_error);
  EXPECT_THROW(rising_factorial(2, c), std::domain_error);
  EXPECT_THROW(rising_factorial(c, c), std::domain_error);

  x = rising_factorial(a,a);
  EXPECT_FLOAT_EQ(840.0, x.val_);
  EXPECT_FLOAT_EQ(840.0 * (2 * digamma(8) - digamma(4)), x.d_);

  x = rising_factorial(5, a);
  EXPECT_FLOAT_EQ(1680.0, x.val_);
  EXPECT_FLOAT_EQ(1680.0 * digamma(9), x.d_);
}
