#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdOperatorLessThanOrEqual,Fvar) {
  using stan::agrad::fvar;
  fvar<double> v4 = 4;
  fvar<double> v5 = 5;
  double d4 = 4;
  double d5 = 5;
  
  EXPECT_TRUE(v4 <= v5);
  EXPECT_TRUE(v4 <= d5);
  EXPECT_TRUE(d4 <= v5);
  EXPECT_TRUE(d4 <= d5);

  int i4 = 4;
  int i5 = 5;
  int i6 = 5;

  EXPECT_TRUE(v4 <= v5);
  EXPECT_TRUE(v4 <= i5);
  EXPECT_TRUE(i4 <= v5);
  EXPECT_TRUE(i4 <= i5);
  EXPECT_TRUE(i4 <= d5);
  EXPECT_TRUE(d4 <= i5);
  EXPECT_TRUE(i5 <= i6);
  EXPECT_TRUE(i5 <= d5);
  EXPECT_FALSE(i5 <= v4);
}

TEST(AgradFwdOperatorLessThanOrEqual, FvarVar) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  fvar<var> y(1.5,1.0);
  fvar<var> z(0.5,1.3);

  EXPECT_TRUE(z <= x);
  EXPECT_TRUE(x <= y);
  EXPECT_FALSE(y <= z);
}

TEST(AgradFwdOperatorLessThanOrEqual, FvarFvarDouble) {
  using stan::agrad::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > z;
  z.val_.val_ = 0.5;
  z.val_.d_ = 0.0;
  z.d_.val_ = 1.0;
  z.d_.d_ = 0.0;

  EXPECT_TRUE(y <= x);
  EXPECT_TRUE(z <= x);
  EXPECT_TRUE(y <= z);
}
TEST(AgradFwdOperatorLessThanOrEqual, FvarFvarVar) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z;
  z.val_.val_ = 0.5;
  z.val_.d_ = 0.0;
  z.d_.val_ = 1.0;
  z.d_.d_ = 0.0;

  EXPECT_TRUE(y <= x);
  EXPECT_TRUE(z <= x);
  EXPECT_TRUE(y <= z);
}

