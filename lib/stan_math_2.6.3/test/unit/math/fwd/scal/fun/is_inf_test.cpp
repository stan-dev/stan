#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/is_inf.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>

TEST(AgradFwdIsInf,Fvar) {
  using stan::math::fvar;
  using stan::math::is_inf;

  double infinity = std::numeric_limits<double>::infinity();
  double nan = std::numeric_limits<double>::quiet_NaN();
  double min = std::numeric_limits<double>::min();
  double max = std::numeric_limits<double>::max();

  fvar<double> a(infinity,infinity);
  fvar<double> b(max,max);
  fvar<double> c(min,min);
  fvar<double> d(0.5,1.0);
  fvar<double> e(nan,nan);

  EXPECT_TRUE(is_inf(a.val_));
  EXPECT_FALSE(is_inf(b.val_));
  EXPECT_FALSE(is_inf(c.val_));
  EXPECT_FALSE(is_inf(d.val_));
  EXPECT_FALSE(is_inf(e.val_));
}

TEST(AgradFwdIsInf,FvarFvar) {
  using stan::math::fvar;
  using stan::math::is_inf;

  double infinity = std::numeric_limits<double>::infinity();
  double nan = std::numeric_limits<double>::quiet_NaN();
  double min = std::numeric_limits<double>::min();
  double max = std::numeric_limits<double>::max();

  fvar<fvar<double> > a(infinity,infinity);
  fvar<fvar<double> > b(max,max);
  fvar<fvar<double> > c(min,min);
  fvar<fvar<double> > d(0.5,1.0);
  fvar<fvar<double> > e(nan,nan);

  EXPECT_TRUE(is_inf(a.val_.val_));
  EXPECT_FALSE(is_inf(b.val_.val_));
  EXPECT_FALSE(is_inf(c.val_.val_));
  EXPECT_FALSE(is_inf(d.val_.val_));
  EXPECT_FALSE(is_inf(e.val_.val_));
}
