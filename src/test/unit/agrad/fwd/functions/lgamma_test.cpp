#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdLgamma,Fvar) {
  using stan::agrad::fvar;
  using boost::math::lgamma;
  using boost::math::digamma;

  fvar<double> x(0.5,1.0);

  fvar<double> a = lgamma(x);
  EXPECT_FLOAT_EQ(lgamma(0.5), a.val_);
  EXPECT_FLOAT_EQ(digamma(0.5), a.d_);
}

TEST(AgradFwdLgamma,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::lgamma;
  using boost::math::digamma;

  fvar<var> x(0.5,1.3);
  fvar<var> a = lgamma(x);

  EXPECT_FLOAT_EQ(lgamma(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * digamma(0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(digamma(0.5), g[0]);
}
TEST(AgradFwdLgamma,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::lgamma;
  using boost::math::digamma;

  fvar<var> x(0.5,1.3);
  fvar<var> a = lgamma(x);

  EXPECT_FLOAT_EQ(lgamma(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * digamma(0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * 4.9348022, g[0]);
}
TEST(AgradFwdLgamma,FvarFvarDouble) {
  using stan::agrad::fvar;
  using boost::math::lgamma;
  using boost::math::digamma;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = lgamma(x);

  EXPECT_FLOAT_EQ(lgamma(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(digamma(0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = lgamma(y);
  EXPECT_FLOAT_EQ(lgamma(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(digamma(0.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdLgamma,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::lgamma;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = lgamma(x);

  EXPECT_FLOAT_EQ(lgamma(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(digamma(0.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(digamma(0.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = lgamma(y);
  EXPECT_FLOAT_EQ(lgamma(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(digamma(0.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(digamma(0.5), r[0]);
}
TEST(AgradFwdLgamma,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::lgamma;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = lgamma(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(4.9348022, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = lgamma(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(4.9348022, r[0]);
}
TEST(AgradFwdLgamma,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = lgamma(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-16.8287966442343199955963342612, g[0]);
}

struct lgamma_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return lgamma(arg1);
  }
};

TEST(AgradFwdLgamma,lgamma_NaN) {
  lgamma_fun lgamma_;
  test_nan(lgamma_,false);
}
