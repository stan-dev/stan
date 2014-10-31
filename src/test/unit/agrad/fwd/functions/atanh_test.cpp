#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <boost/math/special_functions/atanh.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdAtanh,Fvar) {
  using stan::agrad::fvar;
  using boost::math::atanh;

  fvar<double> x(0.5,1.0);

  fvar<double> a = atanh(x);
  EXPECT_FLOAT_EQ(atanh(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 - 0.5 * 0.5), a.d_);

  fvar<double> y(-0.9,1.0);

  fvar<double> b = atanh(y);
  EXPECT_FLOAT_EQ(atanh(-0.9), b.val_);
  EXPECT_FLOAT_EQ(1 / (1 - 0.9 * 0.9), b.d_);
}

TEST(AgradFwdAtanh,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::atanh;

  fvar<var> x(0.5,1.3);
  fvar<var> a = atanh(x);

  EXPECT_FLOAT_EQ(atanh(0.5), a.val_.val()); 
  EXPECT_FLOAT_EQ(1.3 / (1.0 - 0.5 * 0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / (1.0 - 0.5 * 0.5), g[0]);
}

TEST(AgradFwdAtanh,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::atanh;

  fvar<var> x(0.5,1.3);
  fvar<var> a = atanh(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * 1.7777778, g[0]);
}

TEST(AgradFwdAtanh,FvarFvarDouble) {
  using stan::agrad::fvar;
  using boost::math::atanh;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  fvar<fvar<double> > a = atanh(x);

  EXPECT_FLOAT_EQ(atanh(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(1.0 / (1.0 - 0.5 * 0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = atanh(y);

  EXPECT_FLOAT_EQ(atanh(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(1.0 / (1.0 - 0.5 * 0.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

TEST(AgradFwdAtanh,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::atanh;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > a = atanh(x);

  EXPECT_FLOAT_EQ(atanh(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.0 / (1.0 - 0.5 * 0.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0 / (1.0 - 0.5 * 0.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = atanh(y);

  EXPECT_FLOAT_EQ(atanh(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(1.0 / (1.0 - 0.5 * 0.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.0 / (1.0 - 0.5 * 0.5), r[0]);
}

TEST(AgradFwdAtanh,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::atanh;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > a = atanh(x);

  EXPECT_FLOAT_EQ(atanh(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.0 / (1.0 - 0.5 * 0.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(1.7777778, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = atanh(y);

  EXPECT_FLOAT_EQ(atanh(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(1.0 / (1.0 - 0.5 * 0.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.7777778, r[0]);
}
TEST(AgradFwdAtanh,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using boost::math::atanh;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;
  fvar<fvar<var> > a = atanh(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(8.29629629629629629629629629630, g[0]);
}

struct atanh_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return atanh(arg1);
  }
};

TEST(AgradFwdAtanh,atanh_NaN) {
  atanh_fun atanh_;
  test_nan(atanh_,false);
}
