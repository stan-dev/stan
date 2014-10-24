#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdModifiedBesselSecondKind,Fvar) {
  using stan::agrad::fvar;
  using stan::agrad::modified_bessel_second_kind;

  fvar<double> a(4.0,1.0);
  int b = 1;
  fvar<double> x = modified_bessel_second_kind(b,a);
  EXPECT_FLOAT_EQ(0.0124834988872684314, x.val_);
  EXPECT_FLOAT_EQ(-0.014280550807670132, x.d_);

  fvar<double> c(-3.0,1.0);
  EXPECT_THROW(modified_bessel_second_kind(1, c), std::domain_error);
  EXPECT_THROW(modified_bessel_second_kind(-1, c), std::domain_error);
}

TEST(AgradFwdModifiedBesselSecondKind,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::modified_bessel_second_kind;

  fvar<var> z(4.0,2.0);
  fvar<var> a = modified_bessel_second_kind(1,z);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(2.0 * 0.016833855, g[0]);
}
TEST(AgradFwdModifiedBesselSecondKind,FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::modified_bessel_second_kind;

  fvar<fvar<double> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = modified_bessel_second_kind(1,y);

  EXPECT_FLOAT_EQ(modified_bessel_second_kind(1,4.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(-0.014280550807670132, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > b = modified_bessel_second_kind(1, x);

  EXPECT_FLOAT_EQ(modified_bessel_second_kind(1,4.0), b.val_.val_);
  EXPECT_FLOAT_EQ(-0.014280550807670132, b.val_.d_);
  EXPECT_FLOAT_EQ(0, b.d_.val_);
  EXPECT_FLOAT_EQ(0, b.d_.d_);
}
TEST(AgradFwdModifiedBesselSecondKind,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;

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
  test_nan(modified_bessel_second_kind_,false);
}
