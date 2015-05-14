#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/asin.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/asin.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/fwd/core.hpp>



TEST(AgradFwdAsin,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::asin;

  fvar<var> x(0.5,0.3);
  fvar<var> a = asin(x);

  EXPECT_FLOAT_EQ(asin(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(0.3 / sqrt(1.0 - 0.5 * 0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / sqrt(1.0 - 0.5 * 0.5), g[0]);
}

TEST(AgradFwdAsin,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::asin;

  fvar<var> x(0.5,0.3);
  fvar<var> a = asin(x);

  EXPECT_FLOAT_EQ(asin(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(0.3 / sqrt(1.0 - 0.5 * 0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.3 * 0.76980033, g[0]);
}



TEST(AgradFwdAsin,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::asin;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = asin(x);

  EXPECT_FLOAT_EQ(asin(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 - 0.5 * 0.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0 / sqrt(1.0 - 0.5 * 0.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = asin(y);
  EXPECT_FLOAT_EQ(asin(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 - 0.5 * 0.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.0 / sqrt(1.0 - 0.5 * 0.5), r[0]);
}

TEST(AgradFwdAsin,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::asin;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = asin(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(2.0 * 0.76980033, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = asin(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.0 * 0.76980033, r[0]);
}
TEST(AgradFwdAsin,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::asin;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = asin(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(3.07920143567800, g[0]);
}
struct asin_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return asin(arg1);
  }
};

TEST(AgradFwdAsin,asin_NaN) {
  asin_fun asin_;
  test_nan_mix(asin_,false);
}
