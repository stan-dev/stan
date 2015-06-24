#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradMixOperatorNotEqual, FvarVar) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(0.5,1.3);
  fvar<var> y(1.5,1.0);
  fvar<var> z(0.5,1.3);

  EXPECT_FALSE(x != z);
  EXPECT_TRUE(x != y);
  EXPECT_TRUE(z != y);
}
TEST(AgradMixOperatorNotEqual, FvarFvarVar) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > z;
  z.val_.val_ = 0.5;
  z.d_.val_ = 1.0;

  EXPECT_TRUE(x != y);
  EXPECT_TRUE(x != z);
  EXPECT_FALSE(z != y);
}


TEST(AgradMixOperatorNotEqual, ne_nan) {
  using stan::math::fvar;
  using stan::math::var;
  double nan = std::numeric_limits<double>::quiet_NaN();
  double a = 3.0;
  fvar<var> nan_fv = std::numeric_limits<double>::quiet_NaN();
  fvar<var> a_fv = 3.0;
  fvar<fvar<var> > nan_ffv = std::numeric_limits<double>::quiet_NaN();
  fvar<fvar<var> > a_ffv = 3.0;

  EXPECT_TRUE(a != nan_fv);
  EXPECT_TRUE(a_fv != nan_fv);
  EXPECT_TRUE(nan != nan_fv);
  EXPECT_TRUE(nan_fv != nan_fv);
  EXPECT_TRUE(a_fv != nan);
  EXPECT_TRUE(nan_fv != nan);
  EXPECT_TRUE(nan_fv != a);
  EXPECT_TRUE(nan_fv != a_fv);
  EXPECT_TRUE(nan != a_fv);

  EXPECT_TRUE(a != nan_ffv);
  EXPECT_TRUE(a_ffv != nan_ffv);
  EXPECT_TRUE(nan != nan_ffv);
  EXPECT_TRUE(nan_ffv != nan_ffv);
  EXPECT_TRUE(a_ffv != nan);
  EXPECT_TRUE(nan_ffv != nan);
  EXPECT_TRUE(nan_ffv != a);
  EXPECT_TRUE(nan_ffv != a_ffv);
  EXPECT_TRUE(nan != a_ffv);
}
