#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/acos.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/acos.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/fwd/core.hpp>


TEST(AgradFwdAcos,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::acos;

  fvar<var> x(0.5,0.3);
  fvar<var> a = acos(x);

  EXPECT_FLOAT_EQ(acos(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(-0.3 / sqrt(1.0 - 0.5 * 0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.0 / sqrt(1.0 - 0.5 * 0.5), g[0]);
}
TEST(AgradFwdAcos,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::acos;

  fvar<var> x(0.5,0.3);
  fvar<var> a = acos(x);

  AVEC z = createAVEC(x.val_);
  VEC h;
  a.d_.grad(z,h);
  EXPECT_FLOAT_EQ(-0.5 * 0.3 / (sqrt(1.0 - 0.5 * 0.5) * 0.75), h[0]);
}


TEST(AgradFwdAcos,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::acos;

  fvar<fvar<var> > z;
  z.val_.val_ = 0.5;
  z.val_.d_ = 2.0;

  fvar<fvar<var> > b = acos(z);

  EXPECT_FLOAT_EQ(acos(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(-2.0 / sqrt(1.0 - 0.5 * 0.5), b.val_.d_.val());
  EXPECT_FLOAT_EQ(0, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC y = createAVEC(z.val_.val_);
  VEC g;
  b.val_.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.0 / sqrt(1.0 - 0.5 * 0.5), g[0]);

  fvar<fvar<var> > w;
  w.val_.val_ = 0.5;
  w.d_.val_ = 2.0;

  b = acos(w);
  EXPECT_FLOAT_EQ(acos(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(-2.0 / sqrt(1.0 - 0.5 * 0.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC p = createAVEC(w.val_.val_);
  VEC q;
  b.val_.val_.grad(p,q);
  EXPECT_FLOAT_EQ(-1.0 / sqrt(1.0 - 0.5 * 0.5), q[0]);
}
TEST(AgradFwdAcos,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::acos;

  fvar<fvar<var> > z;
  z.val_.val_ = 0.5;
  z.val_.d_ = 2.0;

  fvar<fvar<var> > b = acos(z);

  AVEC y = createAVEC(z.val_.val_);
  VEC g;
  b.val_.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.5 * 2.0 / (sqrt(1.0 - 0.5 * 0.5) * 0.75), g[0]);

  fvar<fvar<var> > w;
  w.val_.val_ = 0.5;
  w.d_.val_ = 2.0;

  fvar<fvar<var> > c = acos(w);

  AVEC p = createAVEC(w.val_.val_);
  VEC q;
  c.d_.val_.grad(p,q);
  EXPECT_FLOAT_EQ(-0.5 * 2.0 / (sqrt(1.0 - 0.5 * 0.5) * 0.75), q[0]);
}

TEST(AgradFwdAcos,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::acos;

  fvar<fvar<var> > z;
  z.val_.val_ = 0.5;
  z.val_.d_ = 1.0;
  z.d_.val_ = 1.0;

  fvar<fvar<var> > b = acos(z);

  AVEC y = createAVEC(z.val_.val_);
  VEC g;
  b.d_.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-3.07920143567800, g[0]);
}

struct acos_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return acos(arg1);
  }
};

TEST(AgradFwdAcos,acos_NaN) {
  acos_fun acos_;
  test_nan_mix(acos_,false);
}
