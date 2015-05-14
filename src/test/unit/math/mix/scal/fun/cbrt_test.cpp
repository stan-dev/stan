#include <gtest/gtest.h>
#include <boost/math/special_functions/cbrt.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/rev/mat/fun/Eigen_NumTraits.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/cbrt.hpp>
#include <stan/math/rev/scal/fun/cbrt.hpp>



TEST(AgradFwdCbrt,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::cbrt;

  fvar<var> x(1.5,1.3);
  fvar<var> a = cbrt(x);

  EXPECT_FLOAT_EQ(cbrt(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / (3 * cbrt(1.5) * cbrt(1.5)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / (3.0 * cbrt(1.5) * cbrt(1.5)), g[0]);
}

TEST(AgradFwdCbrt,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::cbrt;

  fvar<var> x(1.5,1.3);
  fvar<var> a = cbrt(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-2.0 * 1.3 / 3.0 / (3.0 * cbrt(1.5) * cbrt(1.5) * 1.5), g[0]);
}



TEST(AgradFwdCbrt,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::cbrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = cbrt(x);

  EXPECT_FLOAT_EQ(cbrt(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(2.0 / (3.0 * cbrt(1.5) * cbrt(1.5)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.0 / (3.0 * cbrt(1.5) * cbrt(1.5)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = cbrt(y);
  EXPECT_FLOAT_EQ(cbrt(1.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(2.0 / (3.0 * cbrt(1.5) * cbrt(1.5)), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());


  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(1.0 / (3.0 * cbrt(1.5) * cbrt(1.5)), r[0]);
}

TEST(AgradFwdCbrt,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::cbrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = cbrt(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(2.0 * -2.0 / 3.0 / (3.0 * cbrt(1.5) * cbrt(1.5) * 1.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = cbrt(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.0 * -2.0 / 3.0 / (3.0 * cbrt(1.5) * cbrt(1.5) * 1.5), r[0]);
}
TEST(AgradFwdCbrt,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::cbrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = cbrt(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.12562021866154533528757664877253, g[0]);
}

struct cbrt_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return cbrt(arg1);
  }
};

TEST(AgradFwdCbrt,cbrt_NaN) {
  cbrt_fun cbrt_;
  test_nan_mix(cbrt_,false);
}
