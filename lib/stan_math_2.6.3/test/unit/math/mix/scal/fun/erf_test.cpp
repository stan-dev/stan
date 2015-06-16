#include <gtest/gtest.h>
#include <boost/math/special_functions/erf.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/erf.hpp>
#include <stan/math/rev/scal/fun/erf.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>



TEST(AgradFwdErf,FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::exp;
  using std::sqrt;
  using boost::math::erf;

  fvar<var> x(0.5,1.3);
  fvar<var> a = erf(x);

  EXPECT_FLOAT_EQ(erf(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * 2 * exp(-0.5 * 0.5) / 
                  sqrt(boost::math::constants::pi<double>()), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(2 * exp(-0.5 * 0.5) / 
                  sqrt(boost::math::constants::pi<double>()), g[0]);
}
TEST(AgradFwdErf,FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::exp;
  using std::sqrt;
  using boost::math::erf;

  fvar<var> x(0.5,1.3);
  fvar<var> a = erf(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.3 * 2 * exp(-0.5 * 0.5) / 
                  sqrt(boost::math::constants::pi<double>()), g[0]);
}



TEST(AgradFwdErf,FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::exp;
  using std::sqrt;
  using boost::math::erf;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = erf(x);

  EXPECT_FLOAT_EQ(erf(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(2 * exp(-0.5 * 0.5) / 
                  sqrt(boost::math::constants::pi<double>()), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(2 * exp(-0.5 * 0.5) / 
                  sqrt(boost::math::constants::pi<double>()), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = erf(y);
  EXPECT_FLOAT_EQ(erf(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(2 * exp(-0.5 * 0.5) / 
                  sqrt(boost::math::constants::pi<double>()), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(2 * exp(-0.5 * 0.5) / 
                  sqrt(boost::math::constants::pi<double>()), r[0]);
}

TEST(AgradFwdErf,FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::exp;
  using std::sqrt;
  using boost::math::erf;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = erf(x);

  EXPECT_FLOAT_EQ(erf(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(2 * exp(-0.5 * 0.5) / 
                  sqrt(boost::math::constants::pi<double>()), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-2 * exp(-0.5 * 0.5) / 
                  sqrt(boost::math::constants::pi<double>()), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = erf(y);
  EXPECT_FLOAT_EQ(erf(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(2 * exp(-0.5 * 0.5) / 
                  sqrt(boost::math::constants::pi<double>()), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-2 * exp(-0.5 * 0.5) / 
                  sqrt(boost::math::constants::pi<double>()), r[0]);
}
TEST(AgradFwdErf,FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using std::exp;
  using std::sqrt;
  using boost::math::erf;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = erf(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.878782578935444794093723954824, g[0]);
}

struct erf_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return erf(arg1);
  }
};

TEST(AgradFwdErf,erf_NaN) {
  erf_fun erf_;
  test_nan_mix(erf_,false);
}
