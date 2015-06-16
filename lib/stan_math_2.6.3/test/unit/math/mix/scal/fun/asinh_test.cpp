#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>
#include <boost/math/special_functions/asinh.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/asinh.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/asinh.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/fwd/core.hpp>

class AgradFwdAsinh : public testing::Test {
  void SetUp() {
    stan::math::recover_memory();
  }
};



TEST_F(AgradFwdAsinh,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::asinh;

  fvar<var> x(1.5,1.3);
  fvar<var> a = asinh(x);

  EXPECT_FLOAT_EQ(asinh(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / sqrt(1.0 + 1.5 * 1.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 / sqrt(1.0 + 1.5 * 1.5), g[0]);
}

TEST_F(AgradFwdAsinh,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::asinh;

  fvar<var> x(1.5,1.3);
  fvar<var> a = asinh(x);

  EXPECT_FLOAT_EQ(asinh(1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / sqrt(1.0 + 1.5 * 1.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * -0.25601548, g[0]);
}



TEST_F(AgradFwdAsinh,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::asinh;

  stan::math::recover_memory();

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = asinh(x);

  EXPECT_FLOAT_EQ(asinh(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 + 1.5 * 1.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  stan::math::recover_memory();
  EXPECT_FLOAT_EQ(1.0 / sqrt(1.0 + 1.5 * 1.5), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = asinh(y);
  EXPECT_FLOAT_EQ(asinh(1.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(2.0 / sqrt(1.0 + 1.5 * 1.5), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  stan::math::recover_memory();
  EXPECT_FLOAT_EQ(1.0 / sqrt(1.0 + 1.5 * 1.5), r[0]);
}

TEST_F(AgradFwdAsinh,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::asinh;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 2.0;

  fvar<fvar<var> > a = asinh(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(2.0 * -0.25601548, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.5;
  y.d_.val_ = 2.0;

  fvar<fvar<var> > b = asinh(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2.0 * -0.25601548, r[0]);
}
TEST_F(AgradFwdAsinh,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::asinh;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = asinh(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.183805982181141, g[0]);
}
struct asinh_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return asinh(arg1);
  }
};

TEST_F(AgradFwdAsinh,asinh_NaN) {
  asinh_fun asinh_;
  test_nan_mix(asinh_,false);
}
