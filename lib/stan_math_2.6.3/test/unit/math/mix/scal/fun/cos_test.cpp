#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/rev/scal/fun/cos.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>



TEST(AgradFwdCos,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::sin;
  using std::cos;

  fvar<var> x(1.5,1.3);
  fvar<var> a = cos(x);

  EXPECT_FLOAT_EQ(cos(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * -sin(1.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-sin(1.5), g[0]);
}

TEST(AgradFwdCos,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::sin;
  using std::cos;

  fvar<var> x(1.5,1.3);
  fvar<var> a = cos(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.3 * cos(1.5), g[0]);
}


TEST(AgradFwdCos,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::sin;
  using std::cos;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = cos(x);

  EXPECT_FLOAT_EQ(cos(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-2.0 * sin(1.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-sin(1.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = cos(y);
  EXPECT_FLOAT_EQ(cos(1.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(-2.0 * sin(1.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-sin(1.5), r[0]);
}
TEST(AgradFwdCos,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::sin;
  using std::cos;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = cos(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-2.0 * cos(1.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = cos(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-2.0 * cos(1.5), r[0]);
}
TEST(AgradFwdCos,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::sin;
  using std::cos;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = cos(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.99749498660405443094172337114149, g[0]);
}

struct cos_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return cos(arg1);
  }
};

TEST(AgradFwdCos,cos_NaN) {
  cos_fun cos_;
  test_nan_mix(cos_,false);
}
