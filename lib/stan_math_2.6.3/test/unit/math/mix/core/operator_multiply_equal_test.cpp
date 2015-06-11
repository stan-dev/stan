#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradMixOperatorMultiplyEqual, FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(0.5,1.3);
  fvar<var> z(0.5,1.3);
  x *= z;

  EXPECT_FLOAT_EQ(0.25, x.val_.val());
  EXPECT_FLOAT_EQ(1.3 * 0.5 + 1.3 * 0.5, x.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  x.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1, g[0]);
  EXPECT_FLOAT_EQ(0.5, g[1]);
}
TEST(AgradMixOperatorMultiplyEqual, FvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(0.5,1.3);
  double z(0.5);
  x *= z;

  EXPECT_FLOAT_EQ(0.25, x.val_.val());
  EXPECT_FLOAT_EQ(1.3 * 0.5, x.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  x.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1, g[0]);
}
TEST(AgradMixOperatorMultiplyEqual, FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(0.5,1.3);
  fvar<var> z(0.5,1.3);
  x *= z;

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  x.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(1.3, g[1]);
}
TEST(AgradMixOperatorMultiplyEqual, FvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(0.5,1.3);
  double z(0.5);
  x *= z;

  AVEC y = createAVEC(x.val_);
  VEC g;
  x.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradMixOperatorMultiplyEqual, FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  x *= y;
  EXPECT_FLOAT_EQ(0.25, x.val_.val_.val());
  EXPECT_FLOAT_EQ(0.5, x.val_.d_.val());
  EXPECT_FLOAT_EQ(0.5, x.d_.val_.val());
  EXPECT_FLOAT_EQ(1, x.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  x.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
  EXPECT_FLOAT_EQ(0.5, g[1]);
}
TEST(AgradMixOperatorMultiplyEqual, FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  double y(0.5);

  x *= y;
  EXPECT_FLOAT_EQ(0.25, x.val_.val_.val());
  EXPECT_FLOAT_EQ(1 * 0.5, x.val_.d_.val());
  EXPECT_FLOAT_EQ(0, x.d_.val_.val());
  EXPECT_FLOAT_EQ(0, x.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}
TEST(AgradMixOperatorMultiplyEqual, FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  x *= y;

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  x.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(1, g[1]);
}
TEST(AgradMixOperatorMultiplyEqual, FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  x *= y;

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  x.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(0, g[1]);
}
TEST(AgradMixOperatorMultiplyEqual, FvarFvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  double y(0.5);

  x *= y;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradMixOperatorMultiplyEqual, FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  x *= y;

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  x.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(0, g[1]);
}
TEST(AgradMixOperatorMultiplyEqual, FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;
  double y(0.5);

  x *= y;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}


TEST(AgradMixOperatorMultiplyEqual, mult_eq_nan) {
  using stan::math::fvar;
  using stan::math::var;
  double nan = std::numeric_limits<double>::quiet_NaN();
  double a = 3.0;
  fvar<var> nan_fv = std::numeric_limits<double>::quiet_NaN();
  fvar<var> a_fv = 3.0;
  fvar<fvar<var> > nan_ffv = std::numeric_limits<double>::quiet_NaN();
  fvar<fvar<var> > a_ffv = 3.0;

  EXPECT_TRUE(boost::math::isnan( (nan_fv*=a).val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_fv*=a_fv).val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_fv*=nan).val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_fv*=nan_fv).val().val()));
  EXPECT_TRUE(boost::math::isnan( (a_fv*=nan).val().val()));
  EXPECT_TRUE(boost::math::isnan( (a_fv*=nan_fv).val().val()));

  EXPECT_TRUE(boost::math::isnan( (nan_ffv*=a).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_ffv*=a_ffv).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_ffv*=nan).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_ffv*=nan_ffv).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (a_ffv*=nan).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (a_ffv*=nan_ffv).val().val().val()));
}
