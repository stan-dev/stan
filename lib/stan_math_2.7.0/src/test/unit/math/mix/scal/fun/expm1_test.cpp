#include <gtest/gtest.h>
#include <boost/math/special_functions/expm1.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/expm1.hpp>
#include <stan/math/rev/scal/fun/expm1.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>



TEST(AgradFwdExpm1,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::expm1;
  using std::exp;

  fvar<var> x(0.5,1.3);
  fvar<var> a = expm1(x);

  EXPECT_FLOAT_EQ(expm1(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * exp(0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(0.5), g[0]);
}

TEST(AgradFwdExpm1,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::expm1;
  using std::exp;

  fvar<var> x(0.5,1.3);
  fvar<var> a = expm1(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * exp(0.5), g[0]);
}

TEST(AgradFwdExpm1,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::expm1;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = expm1(x);

  EXPECT_FLOAT_EQ(expm1(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp(0.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(0.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = expm1(y);
  EXPECT_FLOAT_EQ(expm1(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(exp(0.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(exp(0.5), r[0]);
}


TEST(AgradFwdExpm1,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::expm1;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = expm1(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(0.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = expm1(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(exp(0.5), r[0]);
}
TEST(AgradFwdExpm1,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = expm1(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(0.5), g[0]);
}

struct expm1_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return expm1(arg1);
  }
};

TEST(AgradFwdExpm1,expm1_NaN) {
  expm1_fun expm1_;
  test_nan_mix(expm1_,false);
}

