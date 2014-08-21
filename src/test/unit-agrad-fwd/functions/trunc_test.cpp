#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdTrunc, Fvar) {
  using stan::agrad::fvar;
  using boost::math::trunc;

  fvar<double> x(0.5,1.0);
  fvar<double> y(2.4,2.0);

  fvar<double> a = trunc(x);
  EXPECT_FLOAT_EQ(trunc(0.5), a.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_);

  fvar<double> b = trunc(y);
  EXPECT_FLOAT_EQ(trunc(2.4), b.val_);
  EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c = trunc(2 * x);
  EXPECT_FLOAT_EQ(trunc(2 * 0.5), c.val_);
  EXPECT_FLOAT_EQ(0.0, c.d_);
}

TEST(AgradFwdTrunc, FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::trunc;

  fvar<var> x(1.5,1.3);
  fvar<var> a = trunc(x);

  EXPECT_FLOAT_EQ(trunc(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdTrunc, FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::trunc;

  fvar<var> x(1.5,1.3);
  fvar<var> a = trunc(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFwdTrunc, FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::trunc;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = trunc(x);

  EXPECT_FLOAT_EQ(trunc(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = trunc(y);
  EXPECT_FLOAT_EQ(trunc(1.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(0, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}
TEST(AgradFwdTrunc, FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::trunc;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = trunc(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = trunc(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0, r[0]);
}
TEST(AgradFwdTrunc, FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::trunc;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = trunc(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}
TEST(AgradFwdTrunc, FvarFvarDouble) {
  using stan::agrad::fvar;
  using boost::math::trunc;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<double> > a = trunc(x);

  EXPECT_FLOAT_EQ(trunc(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  a = trunc(y);
  EXPECT_FLOAT_EQ(trunc(1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

TEST(AgradFwdTrunc,nan) {
  stan::agrad::fvar<double> nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::agrad::trunc(nan).val());
}
