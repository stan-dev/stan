#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/sinh.hpp>
#include <stan/math/rev/scal/fun/sinh.hpp>

TEST(AgradFwdSqrt, FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::sqrt;

  fvar<var> x(1.5,1.3);
  fvar<var> a = sqrt(x);

  EXPECT_FLOAT_EQ(sqrt(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * 0.5 / sqrt(1.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0.5 / sqrt(1.5), g[0]);
}
TEST(AgradFwdSqrt, FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::sqrt;

  fvar<var> x(1.5,1.3);
  fvar<var> a = sqrt(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.5 * -0.5 * 1.3 / sqrt(1.5) / 1.5, g[0]);
}

TEST(AgradFwdSqrt, FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::sqrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = sqrt(x);

  EXPECT_FLOAT_EQ(sqrt(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(2.0 * 0.5 / sqrt(1.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.5 / sqrt(1.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = sqrt(y);
  EXPECT_FLOAT_EQ(sqrt(1.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(2.0 * 0.5 / sqrt(1.5), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0.5 / sqrt(1.5), r[0]);
}
TEST(AgradFwdSqrt, FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::sqrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = sqrt(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.5 * 2.0 * -0.5 / 1.5 / sqrt(1.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = sqrt(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0.5 * -2.0 * 0.5 / 1.5 / sqrt(1.5), r[0]);
}
TEST(AgradFwdSqrt, FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::sqrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = sqrt(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.1360827634879543387887380041503, g[0]);
}

struct sqrt_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return sqrt(arg1);
  }
};

TEST(AgradFwdSqrt,sqrt_NaN) {
  sqrt_fun sqrt_;
  test_nan_mix(sqrt_,false);
}
