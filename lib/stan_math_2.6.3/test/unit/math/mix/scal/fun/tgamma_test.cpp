#include <gtest/gtest.h>
#include <stan/math/fwd/scal/fun/tgamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/ceil.hpp>
#include <stan/math/rev/scal/fun/ceil.hpp>
#include <stan/math/fwd/scal/fun/digamma.hpp>
#include <stan/math/rev/scal/fun/digamma.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/rev/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/rev/scal/fun/tan.hpp>
#include <stan/math/fwd/scal/fun/tgamma.hpp>
#include <stan/math/rev/scal/fun/tgamma.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradFwdTgamma, FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::tgamma;
  using boost::math::digamma;

  fvar<var> x(0.5,1.3);
  fvar<var> a = tgamma(x);

  EXPECT_FLOAT_EQ(tgamma(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * tgamma(0.5) * digamma(0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(tgamma(0.5) * digamma(0.5), g[0]);
}
TEST(AgradFwdTgamma, FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::tgamma;
  using boost::math::digamma;

  fvar<var> x(0.5,1.3);
  fvar<var> a = tgamma(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(20.25423, g[0]);
}

TEST(AgradFwdTgamma, FvarFvarDouble) {
  using stan::math::fvar;
  using boost::math::tgamma;
  using boost::math::digamma;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = tgamma(x);

  EXPECT_FLOAT_EQ(tgamma(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(tgamma(0.5) * digamma(0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = tgamma(y);
  EXPECT_FLOAT_EQ(tgamma(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(tgamma(0.5) * digamma(0.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdTgamma, FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::tgamma;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = tgamma(x);

  EXPECT_FLOAT_EQ(tgamma(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(tgamma(0.5) * digamma(0.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(tgamma(0.5) * digamma(0.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = tgamma(y);
  EXPECT_FLOAT_EQ(tgamma(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(tgamma(0.5) * digamma(0.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(tgamma(0.5) * digamma(0.5), r[0]);
}
TEST(AgradFwdTgamma, FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::tgamma;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = tgamma(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(15.580177, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = tgamma(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(15.580177, r[0]);
}
TEST(AgradFwdTgamma, FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::tgamma;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = tgamma(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-94.768602309214783224297691187, g[0]);
}

struct tgamma_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return tgamma(arg1);
  }
};

TEST(AgradFwdTgamma,tgamma_NaN) {
  tgamma_fun tgamma_;
  test_nan_mix(tgamma_,false);
}
