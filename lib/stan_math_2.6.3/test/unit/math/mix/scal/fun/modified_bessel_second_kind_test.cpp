#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/modified_bessel_second_kind.hpp>
#include <stan/math/rev/scal/fun/modified_bessel_second_kind.hpp>

TEST(AgradFwdModifiedBesselSecondKind,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::modified_bessel_second_kind;

  fvar<var> z(4.0,2.0);
  fvar<var> a = modified_bessel_second_kind(1,z);

  EXPECT_FLOAT_EQ(modified_bessel_second_kind(1, 4.0), a.val_.val());
  EXPECT_FLOAT_EQ(-0.014280550807670132 * 2.0, a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.014280550807670132, g[0]);
}
TEST(AgradFwdModifiedBesselSecondKind,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::modified_bessel_second_kind;

  fvar<var> z(4.0,2.0);
  fvar<var> a = modified_bessel_second_kind(1,z);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(2.0 * 0.016833855, g[0]);
}

TEST(AgradFwdModifiedBesselSecondKind,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::modified_bessel_second_kind;

  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = modified_bessel_second_kind(1,y);

  EXPECT_FLOAT_EQ(modified_bessel_second_kind(1,4.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(-0.014280550807670132, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.014280550807670132, g[0]);

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > b = modified_bessel_second_kind(1, x);

  EXPECT_FLOAT_EQ(modified_bessel_second_kind(1,4.0), b.val_.val_.val());
  EXPECT_FLOAT_EQ(-0.014280550807670132, b.val_.d_.val());
  EXPECT_FLOAT_EQ(0, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-0.014280550807670132, r[0]);
}
TEST(AgradFwdModifiedBesselSecondKind,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::modified_bessel_second_kind;

  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = modified_bessel_second_kind(1,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.016833855, g[0]);

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > b = modified_bessel_second_kind(1, x);

  AVEC q = createAVEC(x.val_.val_);
  VEC r;
  b.val_.d_.grad(q,r);
  EXPECT_FLOAT_EQ(0.016833855, r[0]);
}
TEST(AgradFwdModifiedBesselSecondKind,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = modified_bessel_second_kind(1,x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.0206641928162660975059, g[0]);
}

struct modified_bessel_second_kind_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return modified_bessel_second_kind(1,arg1);
  }
};

TEST(AgradFwdModifiedBesselSecondKind,modified_bessel_second_kind_NaN) {
  modified_bessel_second_kind_fun modified_bessel_second_kind_;
  test_nan_mix(modified_bessel_second_kind_,false);
}
