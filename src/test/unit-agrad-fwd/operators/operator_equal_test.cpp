#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdOperatorEqual, Fvar) {
  using stan::agrad::fvar;
  fvar<double> v4 = 4;
  fvar<double> v5 = 5;
  double d4 = 4;
  double d5 = 5;
  
  EXPECT_FALSE(v5 == v4);
  EXPECT_FALSE(d5 == v4);
  EXPECT_FALSE(v5 == d4);
  EXPECT_FALSE(d5 == d4);

  int i4 = 4;
  int i5 = 5;
  int i6 = 5;

  EXPECT_FALSE(i5 == v4);
  EXPECT_FALSE(v5 == i4);
  EXPECT_FALSE(i5 == i4);
  EXPECT_FALSE(d5 == i4);
  EXPECT_FALSE(i5 == d4);
  EXPECT_TRUE(i6 == i5);
  EXPECT_TRUE(i6 == v5);
}

TEST(AgradFwdOperatorEqual, FvarVar) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  fvar<var> y(1.5,1.0);
  fvar<var> z(0.5,1.3);

  EXPECT_TRUE(x == z);
  EXPECT_FALSE(x == y);
  EXPECT_FALSE(z == y);
}

TEST(AgradFwdOperatorEqual, FvarFvarDouble) {
  using stan::agrad::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > z;
  z.val_.val_ = 0.5;
  z.d_.val_ = 1.0;

  EXPECT_FALSE(x == y);
  EXPECT_FALSE(x == z);
  EXPECT_TRUE(z == y);
}

TEST(AgradFwdOperatorEqual, FvarFvarVar) {
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
  z.d_.val_ = 1.0;

  EXPECT_FALSE(x == y);
  EXPECT_FALSE(x == z);
  EXPECT_TRUE(z == y);
}
