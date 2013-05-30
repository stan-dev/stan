#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>

TEST(AgradFvar, falling_factorial) {
  using stan::agrad::fvar;
  using stan::agrad::falling_factorial;
  using boost::math::digamma;

  fvar<double> a(4.0,1.0);
  fvar<double> x = falling_factorial(a,1);
  EXPECT_FLOAT_EQ(24.0, x.val_);
  EXPECT_FLOAT_EQ(24.0 * digamma(5), x.d_);

  fvar<double> c(-3.0,2.0);

  EXPECT_THROW(falling_factorial(c, 2), std::domain_error);
  EXPECT_THROW(falling_factorial(2, c), std::domain_error);
  EXPECT_THROW(falling_factorial(c, c), std::domain_error);

  x = falling_factorial(a,a);
  EXPECT_FLOAT_EQ(1.0, x.val_);
  EXPECT_FLOAT_EQ(0.0, x.d_);

  x = falling_factorial(5, a);
  EXPECT_FLOAT_EQ(5.0, x.val_);
  EXPECT_FLOAT_EQ(-5.0 * digamma(5.0),x.d_);
}
