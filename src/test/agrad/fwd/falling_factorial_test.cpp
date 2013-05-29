#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/digamma.hpp>

TEST(AgradFvar, falling_factorial) {
  using stan::agrad::fvar;
  using stan::agrad::falling_factorial;
  using boost::math::digamma;

  fvar<double> a;
  a.val_ = 4.0;
  a.d_ = 1.0;
  fvar<double> x = falling_factorial(a,1);
  EXPECT_FLOAT_EQ(24.0, x.val_);
  EXPECT_FLOAT_EQ(24.0 * digamma(5), x.d_);

  fvar<double> c;
  c.val_ = -3.0;
  c.d_ = 2.0;

  EXPECT_THROW(falling_factorial(c, 2), std::domain_error);
}
