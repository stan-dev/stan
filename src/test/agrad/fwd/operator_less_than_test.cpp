#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar,lt) {
  using stan::agrad::fvar;
  fvar<double> v4 = 4;
  fvar<double> v5 = 5;
  double d4 = 4;
  double d5 = 5;
  
  EXPECT_TRUE(v4 < v5);
  EXPECT_TRUE(v4 < d5);
  EXPECT_TRUE(d4 < v5);
  EXPECT_TRUE(d4 < d5);

  int i4 = 4;
  int i5 = 5;

  EXPECT_TRUE(v4 < v5);
  EXPECT_TRUE(v4 < i5);
  EXPECT_TRUE(i4 < v5);
  EXPECT_TRUE(i4 < i5);
  EXPECT_TRUE(i4 < d5);
  EXPECT_TRUE(d4 < i5);
}

TEST(AgradFvarVar, lt) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x;
  x.val_ = 0.5;
  x.d_ = 1.3;

  fvar<var> y;
  y.val_ = 1.5;
  y.d_ = 1.0;

  fvar<var> z;
  z.val_ = 0.5;
  z.d_ = 1.3;

  EXPECT_FALSE(z < x);
  EXPECT_FALSE(y < x);
  EXPECT_FALSE(y < z);
}

TEST(AgradFvarFvar, lt) {
  using stan::agrad::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 0.0;
  x.d_.d_ = 0.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.val_.d_ = 0.0;
  y.d_.val_ = 1.0;
  y.d_.d_ = 0.0;

  fvar<fvar<double> > z;
  z.val_.val_ = 0.5;
  z.val_.d_ = 0.0;
  z.d_.val_ = 1.0;
  z.d_.d_ = 0.0;

  EXPECT_TRUE(y < x);
  EXPECT_TRUE(z < x);
  EXPECT_FALSE(y < z);
}
