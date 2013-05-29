#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, rising_factorial) {
  using stan::agrad::fvar;
  using stan::agrad::rising_factorial;

  fvar<double> a;
  a.val_ = 4.0;
  a.d_ = 1.0;
  fvar<double> x = rising_factorial(a,1);
  EXPECT_FLOAT_EQ(4.0, x.val_);
  EXPECT_FLOAT_EQ(1.0, x.d_);

  fvar<double> c;
  c.val_ = -3.0;
  c.d_ = 2.0;

  EXPECT_THROW(rising_factorial(c, 2), std::domain_error);
}
