#include <gtest/gtest.h>
#include <boost/math/special_functions/atanh.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/atanh.hpp>
#include <stan/math/rev/scal/fun/atanh.hpp>
#include <stan/math/fwd/core.hpp>



TEST(AgradFwdAtanh,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::atanh;

  fvar<var> x(0.5,1.3);
  fvar<var> a = atanh(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * 1.7777778, g[0]);
}



TEST(AgradFwdAtanh,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  test_nan_mix(atanh_,false);
}
