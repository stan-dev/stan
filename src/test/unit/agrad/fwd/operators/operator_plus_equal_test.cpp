#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdOperatorPlusEqual, Fvar) {
  using stan::agrad::fvar;

  fvar<double> a(0.5,1.0);
  fvar<double> x1(0.4,2.0);
  a += x1;
  EXPECT_FLOAT_EQ(0.5 + 0.4, a.val_);
  EXPECT_FLOAT_EQ(1.0 + 2.0, a.d_);

  fvar<double> b(0.5,1.0);
  fvar<double> x2(0.4,2.0);
  b += -x2;
  EXPECT_FLOAT_EQ(0.5 - 0.4, b.val_);
  EXPECT_FLOAT_EQ(1.0 - 2.0, b.d_);

  fvar<double> c(0.6,3.0);
  double x3(0.3);
  c += x3;
  EXPECT_FLOAT_EQ(0.6 + 0.3, c.val_);
  EXPECT_FLOAT_EQ(3.0, c.d_);

  fvar<double> d(0.5,1.0);
  fvar<double> x4(-0.4,2.0);
  d += x4;
  EXPECT_FLOAT_EQ(0.5 - 0.4, d.val_);
  EXPECT_FLOAT_EQ(1.0 + 2.0, d.d_);
}

TEST(AgradFwdOperatorPlusEqual, FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  x += 0.3;
  EXPECT_FLOAT_EQ(0.5 + 0.3, x.val_.val());
  EXPECT_FLOAT_EQ(1.3, x.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  x.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}
TEST(AgradFwdOperatorPlusEqual, FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x(0.5,1.3);
  x += 0.3;

  AVEC y = createAVEC(x.val_);
  VEC g;
  x.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFwdOperatorPlusEqual, FvarFvarDouble) {
  using stan::agrad::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  x += 0.3;
  EXPECT_FLOAT_EQ(0.5 + 0.3, x.val_.val_);
  EXPECT_FLOAT_EQ(1, x.val_.d_);
  EXPECT_FLOAT_EQ(0, x.d_.val_);
  EXPECT_FLOAT_EQ(0, x.d_.d_);
}
TEST(AgradFwdOperatorPlusEqual, FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  x += 0.3;
  EXPECT_FLOAT_EQ(0.5 + 0.3, x.val_.val_.val());
  EXPECT_FLOAT_EQ(1, x.val_.d_.val());
  EXPECT_FLOAT_EQ(0, x.d_.val_.val());
  EXPECT_FLOAT_EQ(0, x.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}
TEST(AgradFwdOperatorPlusEqual, FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  x += 0.3;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdOperatorPlusEqual, FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  x += 0.3;

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  x.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFwdOperatorPlusEqual, plus_eq_nan) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  double nan = std::numeric_limits<double>::quiet_NaN();
  double a = 3.0;
  fvar<double> nan_fd = std::numeric_limits<double>::quiet_NaN();
  fvar<double> a_fd = 3.0;
  fvar<var> nan_fv = std::numeric_limits<double>::quiet_NaN();
  fvar<var> a_fv = 3.0;
  fvar<fvar<double> > nan_ffd = std::numeric_limits<double>::quiet_NaN();
  fvar<fvar<double> > a_ffd = 3.0;
  fvar<fvar<var> > nan_ffv = std::numeric_limits<double>::quiet_NaN();
  fvar<fvar<var> > a_ffv = 3.0;

  EXPECT_TRUE(boost::math::isnan( (nan_fd+=a).val()));
  EXPECT_TRUE(boost::math::isnan( (nan_fd+=a_fd).val()));
  EXPECT_TRUE(boost::math::isnan( (nan_fd+=nan).val()));
  EXPECT_TRUE(boost::math::isnan( (nan_fd+=nan_fd).val()));
  EXPECT_TRUE(boost::math::isnan( (a_fd+=nan).val()));
  EXPECT_TRUE(boost::math::isnan( (a_fd+=nan_fd).val()));

  EXPECT_TRUE(boost::math::isnan( (nan_fv+=a).val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_fv+=a_fv).val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_fv+=nan).val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_fv+=nan_fv).val().val()));
  EXPECT_TRUE(boost::math::isnan( (a_fv+=nan).val().val()));
  EXPECT_TRUE(boost::math::isnan( (a_fv+=nan_fv).val().val()));

  EXPECT_TRUE(boost::math::isnan( (nan_ffd+=a).val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_ffd+=a_ffd).val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_ffd+=nan).val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_ffd+=nan_ffd).val().val()));
  EXPECT_TRUE(boost::math::isnan( (a_ffd+=nan).val().val()));
  EXPECT_TRUE(boost::math::isnan( (a_ffd+=nan_ffd).val().val()));

  EXPECT_TRUE(boost::math::isnan( (nan_ffv+=a).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_ffv+=a_ffv).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_ffv+=nan).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (nan_ffv+=nan_ffv).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (a_ffv+=nan).val().val().val()));
  EXPECT_TRUE(boost::math::isnan( (a_ffv+=nan_ffv).val().val().val()));
}
