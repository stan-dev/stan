#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <stan/math/fwd/scal/fun/lbeta.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/digamma.hpp>
#include <stan/math/rev/scal/fun/digamma.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/lgamma.hpp>
#include <stan/math/rev/scal/fun/lgamma.hpp>
#include <stan/math/fwd/scal/fun/lmgamma.hpp>
#include <stan/math/rev/scal/fun/lmgamma.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/rev/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>


TEST(AgradFwdLbeta,FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<var> x(3.0,1.3);
  fvar<var> z(6.0,1.0);
  fvar<var> a = lbeta(x,z);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * digamma(3.0) + digamma(6.0) - (1.0 + 1.3) * 
                  digamma(3.0 + 6.0), a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(digamma(3.0) - digamma(9.0),g[0]);
  EXPECT_FLOAT_EQ(digamma(6.0) - digamma(9.0),g[1]);
}
TEST(AgradFwdLbeta,FvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<var> x(3.0,1.3);
  double z(6.0);
  fvar<var> a = lbeta(x,z);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * digamma(3.0) - (1.3) * 
                  digamma(3.0 + 6.0), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(digamma(3.0) - digamma(9.0),g[0]);
}
TEST(AgradFwdLbeta,Double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  double x(3.0);
  fvar<var> z(6.0,1.0);
  fvar<var> a = lbeta(x,z);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ(digamma(6.0) - digamma(3.0 + 6.0), a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(digamma(6.0) - digamma(9.0),g[0]);
}
TEST(AgradFwdLbeta,FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<var> x(3.0,1.3);
  fvar<var> z(6.0,1.0);
  fvar<var> a = lbeta(x,z);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * digamma(3.0) + digamma(6.0) - (1.0 + 1.3) * 
                  digamma(3.0 + 6.0), a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * 0.39493407 - 2.3 * 0.11751201,g[0]);
  EXPECT_FLOAT_EQ(0.18132296 - 2.3 * 0.11751201,g[1]);
}
TEST(AgradFwdLbeta,FvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<var> x(3.0,1.3);
  double z(6.0);
  fvar<var> a = lbeta(x,z);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * digamma(3.0) - (1.3) * 
                  digamma(3.0 + 6.0), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * 0.39493407 - 1.3 * 0.11751201,g[0]);
}
TEST(AgradFwdLbeta,Double_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  double x(3.0);
  fvar<var> z(6.0,1.0);
  fvar<var> a = lbeta(x,z);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ(digamma(6.0) - digamma(3.0 + 6.0), a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.18132296 - 0.11751201,g[0]);
}
TEST(AgradFwdLbeta,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = lbeta(x,y);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(digamma(3.0) - digamma(9.0), a.val_.d_.val());
  EXPECT_FLOAT_EQ(digamma(6.0) - digamma(9.0), a.d_.val_.val());
  EXPECT_FLOAT_EQ(-0.11751202, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(digamma(3.0) - digamma(9.0), g[0]);
  EXPECT_FLOAT_EQ(digamma(6.0) - digamma(9.0), g[1]);
}
TEST(AgradFwdLbeta,FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  double y(6.0);

  fvar<fvar<var> > a = lbeta(x,y);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(digamma(3.0) - digamma(9.0), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(digamma(3.0) - digamma(9.0), g[0]);
}
TEST(AgradFwdLbeta,Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  double x(3.0);

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = lbeta(x,y);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(digamma(6.0) - digamma(9.0), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(digamma(6.0) - digamma(9.0), g[0]);
}
TEST(AgradFwdLbeta,FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = lbeta(x,y);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(digamma(3.0) - digamma(9.0), a.val_.d_.val());
  EXPECT_FLOAT_EQ(digamma(6.0) - digamma(9.0), a.d_.val_.val());
  EXPECT_FLOAT_EQ(-0.11751202, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.39493407 - 0.11751201, g[0]);
  EXPECT_FLOAT_EQ(-0.11751202, g[1]);
}
TEST(AgradFwdLbeta,FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = lbeta(x,y);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(digamma(3.0) - digamma(9.0), a.val_.d_.val());
  EXPECT_FLOAT_EQ(digamma(6.0) - digamma(9.0), a.d_.val_.val());
  EXPECT_FLOAT_EQ(-0.11751202, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.11751202, g[0]);
  EXPECT_FLOAT_EQ(0.18132296 - 0.11751201, g[1]);
}
TEST(AgradFwdLbeta,FvarFvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  double y(6.0);

  fvar<fvar<var> > a = lbeta(x,y);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(digamma(3.0) - digamma(9.0), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.39493407 - 0.11751201, g[0]);
}
TEST(AgradFwdLbeta,Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  double x(3.0);

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = lbeta(x,y);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(digamma(6.0) - digamma(9.0), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.18132296 - 0.11751201, g[0]);
}
TEST(AgradFwdLbeta,FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = lbeta(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.013793319, g[0]);
  EXPECT_FLOAT_EQ(0.013793319, g[1]);
}
TEST(AgradFwdLbeta,FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  double y(6.0);

  fvar<fvar<var> > a = lbeta(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.140320487123420796890184645287, g[0]);
}
TEST(AgradFwdLbeta,Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::digamma;
  using stan::math::lbeta;

  double x(3.0);

  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = lbeta(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.0189964130493467228161105712126, g[0]);
}


struct lbeta_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return lbeta(arg1,arg2);
  }
};

TEST(AgradFwdLbeta, nan) {
  lbeta_fun lbeta_;
  test_nan_mix(lbeta_,3.0,5.0,false);
}
