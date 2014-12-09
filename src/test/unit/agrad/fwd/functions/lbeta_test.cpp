#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/functions/lbeta.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdLbeta,Fvar) {
  using stan::agrad::fvar;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<double> x(0.5,1.0);
  fvar<double> y(1.2,2.0);

  double w = 1.3;

  fvar<double> a = lbeta(x, y);
  EXPECT_FLOAT_EQ(lbeta(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ(digamma(0.5) + 2.0 * digamma(1.2) 
                  - (1.0 + 2.0) * digamma(0.5 + 1.2), a.d_);

  fvar<double> b = lbeta(x, w);
  EXPECT_FLOAT_EQ(lbeta(0.5, 1.3), b.val_);
  EXPECT_FLOAT_EQ(1.0 * digamma(0.5) - 1.0 * digamma(0.5 + 1.3), b.d_);

  fvar<double> c = lbeta(w, x);
  EXPECT_FLOAT_EQ(lbeta(1.3, 0.5), c.val_);
  EXPECT_FLOAT_EQ(1.0 * digamma(0.5) - 1.0 * digamma(1.3 + 0.5), c.d_);
}

TEST(AgradFwdLbeta,FvarVar_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
TEST(AgradFwdLbeta,FvarFvarDouble) {
  using stan::agrad::fvar;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<fvar<double> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = lbeta(x,y);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(digamma(3.0) - digamma(9.0), a.val_.d_);
  EXPECT_FLOAT_EQ(digamma(6.0) - digamma(9.0), a.d_.val_);
  EXPECT_FLOAT_EQ(-0.11751202, a.d_.d_);
}
TEST(AgradFwdLbeta,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  test_nan(lbeta_,3.0,5.0,false);
}
