#include <gtest/gtest.h>

#include <iostream>

#include <boost/math/special_functions/fpclassify.hpp>

#include <stan/agrad/fvar.hpp>

TEST(AgradFvar,fvar) {
  using stan::agrad::fvar;
  typedef stan::agrad::fvar<double> fvd;

  fvar<double> a;
  EXPECT_FLOAT_EQ(0.0, a.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_);

  fvar<double> b(1.9);
  EXPECT_FLOAT_EQ(1.9, b.val_);
  EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c(1.93, -27.832);
  EXPECT_FLOAT_EQ(1.93, c.val_);
  EXPECT_FLOAT_EQ(-27.832, c.d_);

  fvar<double> d = -c;
  EXPECT_FLOAT_EQ(-1.93, d.val_);
  EXPECT_FLOAT_EQ(27.832, d.d_);

  fvar<double> e(5.0);
  d += e;
  EXPECT_FLOAT_EQ(3.07, d.val_);

  EXPECT_FLOAT_EQ(3.07, (d++).val_);
  EXPECT_FLOAT_EQ(4.07, d.val_);

  EXPECT_FLOAT_EQ(5.07, (++d).val_);
  EXPECT_FLOAT_EQ(5.07, d.val_);

}

TEST(AgradFvar, sin) {
  using stan::agrad::fvar;
  using std::sin;
  using std::cos;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  fvar<double> y = sin(x);

  EXPECT_FLOAT_EQ(sin(0.5), y.val_);
  EXPECT_FLOAT_EQ(cos(0.5), y.d_);

  fvar<double> z = 3 * sin(x) + x;
  EXPECT_FLOAT_EQ(3 * sin(0.5) + 0.5, z.val_);
  EXPECT_FLOAT_EQ(3 * cos(0.5) + 1, z.d_);
                  
  
}


