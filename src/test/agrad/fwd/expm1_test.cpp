#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/expm1.hpp>

TEST(AgradFvar, expm1) {
  using stan::agrad::fvar;
  using boost::math::expm1;
  using std::exp;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = expm1(x);
  EXPECT_FLOAT_EQ(expm1(0.5), a.val_);
  EXPECT_FLOAT_EQ(exp(0.5), a.d_);

  fvar<double> b = 2 * expm1(x) + 4;
  EXPECT_FLOAT_EQ(2 * expm1(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 * exp(0.5), b.d_);

  fvar<double> c = -expm1(x) + 5;
  EXPECT_FLOAT_EQ(-expm1(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-exp(0.5), c.d_);

  fvar<double> d = -3 * expm1(-x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * expm1(-0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(3 * exp(-0.5) + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = expm1(y);
  EXPECT_FLOAT_EQ(expm1(-0.5), e.val_);
  EXPECT_FLOAT_EQ(exp(-0.5), e.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> f = expm1(z);
  EXPECT_FLOAT_EQ(expm1(0.0), f.val_);
  EXPECT_FLOAT_EQ(exp(0.0), f.d_);
}
