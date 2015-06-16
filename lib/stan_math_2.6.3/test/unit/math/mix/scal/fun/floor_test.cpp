#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>



TEST(AgradFwdFloor,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::floor;

  fvar<var> x(1.5,1.3);
  fvar<var> a = floor(x);

  EXPECT_FLOAT_EQ(floor(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFwdFloor,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::floor;

  fvar<var> x(1.5,1.3);
  fvar<var> a = floor(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}



TEST(AgradFwdFloor,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::floor;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = floor(x);

  EXPECT_FLOAT_EQ(floor(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = floor(y);
  EXPECT_FLOAT_EQ(floor(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0.0, r[0]);
}


TEST(AgradFwdFloor,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::floor;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = floor(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.0, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = floor(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0.0, r[0]);
}
TEST(AgradFwdFloor,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = floor(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

struct floor_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return floor(arg1);
  }
};

TEST(AgradFwdFloor,floor_NaN) {
  floor_fun floor_;
  test_nan_mix(floor_,false);
}
