#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/rev/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/digamma.hpp>
#include <stan/math/rev/scal/fun/digamma.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>



TEST(AgradFwdDigamma,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;  
  using boost::math::zeta;
  
  fvar<var> x(0.5,1.3);
  fvar<var> a = digamma(x);

  EXPECT_FLOAT_EQ(digamma(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * 4.9348022005446793094, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(4.9348022005446793094, g[0]);
}

TEST(AgradFwdDigamma,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;  
  using boost::math::zeta;
  
  fvar<var> x(0.5,1.3);
  fvar<var> a = digamma(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * -16.8288, g[0]);
}


TEST(AgradFwdDigamma,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = digamma(x);

  EXPECT_FLOAT_EQ(digamma(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(4.9348022005446793094, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(4.9348022005446793094, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = digamma(y);
  EXPECT_FLOAT_EQ(digamma(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(4.9348022005446793094, b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(4.9348022005446793094, r[0]);
}

TEST(AgradFwdDigamma,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = digamma(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-16.8288, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = digamma(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-16.8288, r[0]);
}
TEST(AgradFwdDigamma,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = digamma(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(97.4090910340024372364403326887, g[0]);
}

struct digamma_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return digamma(arg1);
  }
};

TEST(AgradFwdDigamma,digamma_NaN) {
  digamma_fun digamma_;
  test_nan_mix(digamma_,false);
}
