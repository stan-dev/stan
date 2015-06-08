#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/square.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/square.hpp>
#include <stan/math/rev/scal/fun/square.hpp>

TEST(AgradFwdSquare, FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::square;

  fvar<var> x(1.5,1.3);
  fvar<var> a = square(x);

  EXPECT_FLOAT_EQ(square(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * 2.0 * (1.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(2.0 * (1.5), g[0]);
}
TEST(AgradFwdSquare, FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::square;

  fvar<var> x(1.5,1.3);
  fvar<var> a = square(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * 2.0, g[0]);
}

TEST(AgradFwdSquare, FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::square;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = square(x);

  EXPECT_FLOAT_EQ(square(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(2.0 * 2.0 * (1.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(2.0 * 1.5, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = square(y);
  EXPECT_FLOAT_EQ(square(1.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(2.0 * 2.0 * (1.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.0 * 1.5, r[0]);
}
TEST(AgradFwdSquare, FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::square;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = square(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(2.0 * 2.0, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = square(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.0 * 2.0, r[0]);
}
TEST(AgradFwdSquare, FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::square;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = square(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

struct square_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return square(arg1);
  }
};

TEST(AgradFwdSquare,square_NaN) {
  square_fun square_;
  test_nan_mix(square_,false);
}
