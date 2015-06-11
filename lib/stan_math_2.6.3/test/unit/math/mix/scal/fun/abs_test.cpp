#include <limits>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/core.hpp>


TEST(AgradFwdAbs,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::abs;

  fvar<var> x(2.0,1.0);
  fvar<var> a = abs(x);

  EXPECT_FLOAT_EQ(2.0, a.val_.val());
  EXPECT_FLOAT_EQ(1.0, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}
TEST(AgradFwdAbs,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::abs;

  fvar<var> x(2.0,1.0);
  fvar<var> a = abs(x);

  AVEC z = createAVEC(x.val_);
  VEC h;
  a.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0, h[0]);
}

TEST(AgradFwdAbs,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::abs;

  fvar<fvar<var> > y;
  y.val_ = fvar<var>(4.0,1.0);

  fvar<fvar<var> > b = abs(y);

  EXPECT_FLOAT_EQ(4.0, b.val_.val_.val());
  EXPECT_FLOAT_EQ(1.0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(0.0, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0.0, b.d_.d_.val());

  AVEC z = createAVEC(y.val_.val_);
  VEC h;
  b.val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(1.0, h[0]);
}
TEST(AgradFwdAbs,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::abs;

  fvar<fvar<var> > y;
  y.val_ = fvar<var>(4.0,1.0);

  fvar<fvar<var> > b = abs(y);

  AVEC z = createAVEC(y.val_.val_);
  VEC h;
  b.val_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0, h[0]);
}
TEST(AgradFwdAbs,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::abs;

  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.val_.d_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = abs(y);

  AVEC z = createAVEC(y.val_.val_);
  VEC h;
  b.d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0, h[0]);
}

struct abs_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return abs(arg1);
  }
};

TEST(AgradFwdAbs,abs_NaN) {
  abs_fun abs_;
  test_nan_mix(abs_,false);
}
