#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradMixOperatorMinusEqual, FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(0.5,1.3);
  x -= 0.3;
  EXPECT_FLOAT_EQ(0.5 - 0.3, x.val_.val());
  EXPECT_FLOAT_EQ(1.3, x.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  x.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}
TEST(AgradMixOperatorMinusEqual, FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(0.5,1.3);
  x -= 0.3;

  AVEC y = createAVEC(x.val_);
  VEC g;
  x.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradMixOperatorMinusEqual, FvarFvarDouble) {
  using stan::math::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  x -= 0.3;
  EXPECT_FLOAT_EQ(0.5 - 0.3, x.val_.val_);
  EXPECT_FLOAT_EQ(1, x.val_.d_);
  EXPECT_FLOAT_EQ(0, x.d_.val_);
  EXPECT_FLOAT_EQ(0, x.d_.d_);
}
TEST(AgradMixOperatorMinusEqual, FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  x -= 0.3;
  EXPECT_FLOAT_EQ(0.5 - 0.3, x.val_.val_.val());
  EXPECT_FLOAT_EQ(1, x.val_.d_.val());
  EXPECT_FLOAT_EQ(0, x.d_.val_.val());
  EXPECT_FLOAT_EQ(0, x.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}
TEST(AgradMixOperatorMinusEqual, FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  x -= 0.3;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradMixOperatorMinusEqual, FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  x -= 0.3;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}


TEST(AgradMixOperatorMinusEqual, min_eq_nan) {
  using stan::math::fvar;
  using stan::math::var;
  double nan = std::numeric_limits<double>::quiet_NaN();
  double a = 3.0;
  fvar<var> nan_fv = std::numeric_limits<double>::quiet_NaN();
  fvar<var> a_fv = 3.0;
  fvar<fvar<var> > nan_ffv = std::numeric_limits<double>::quiet_NaN();
  fvar<fvar<var> > a_ffv = 3.0;

  EXPECT_TRUE(boost::math::isnan( (nan_fv-=a).val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_fv-=a_fv).val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_fv-=nan).val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_fv-=nan_fv).val().val()));
  EXPECT_TRUE(boost::math::isnan( (a_fv-=nan).val().val()));
  EXPECT_TRUE(boost::math::isnan( (a_fv-=nan_fv).val().val()));

  EXPECT_TRUE(boost::math::isnan( (nan_ffv-=a).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_ffv-=a_ffv).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_ffv-=nan).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_ffv-=nan_ffv).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (a_ffv-=nan).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (a_ffv-=nan_ffv).val().val().val()));
}
