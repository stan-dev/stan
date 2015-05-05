#include <gtest/gtest.h>
#include <stan/math/fwd/scal/fun/inv.hpp>
#include <stan/math/rev/scal/fun/inv.hpp>
#include <stan/math/prim/scal/fun/inv.hpp>
#include <stan/math/fwd/scal/fun/hypot.hpp>
#include <stan/math/rev/scal/fun/hypot.hpp>
#include <boost/math/special_functions/hypot.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>



TEST(AgradFwdHypot,FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  fvar<var> x(3.0,1.3);

  fvar<var> z(6.0,1.0);
  fvar<var> a = hypot(x,z);

  EXPECT_FLOAT_EQ(hypot(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.3 * 3.0 + 6.0 * 1.0) / hypot(3.0, 6.0), a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(3.0 / hypot(3.0,6.0),g[0]);
  EXPECT_FLOAT_EQ(6.0 / hypot(3.0,6.0),g[1]);
}
TEST(AgradFwdHypot,FvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  fvar<var> x(3.0,1.3);
  double z(6.0);
  fvar<var> a = hypot(x,z);

  EXPECT_FLOAT_EQ(hypot(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((1.3 * 3.0) / hypot(3.0, 6.0), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(3.0 / hypot(3.0,6.0),g[0]);
}
TEST(AgradFwdHypot,Double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  double x(3.0);
  fvar<var> z(6.0,1.0);
  fvar<var> a = hypot(x,z);

  EXPECT_FLOAT_EQ(hypot(3.0,6.0), a.val_.val());
  EXPECT_FLOAT_EQ((6.0 * 1.0) / hypot(3.0, 6.0), a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(6.0 / hypot(3.0,6.0),g[0]);
}
TEST(AgradFwdHypot,FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  fvar<var> x(3.0,1.3);
  fvar<var> z(6.0,1.0);
  fvar<var> a = hypot(x,z);

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ((1.3 * 6.0 * 6.0 - 6.0 * 3.0) 
                  / hypot(3.0,6.0) / (9.0 + 36.0),g[0]);
  EXPECT_FLOAT_EQ((1.0 * 3.0 * 3.0 - 1.3 * 6.0 * 3.0) 
                  / hypot(3.0,6.0) / (9.0 + 36.0),g[1]);
}
TEST(AgradFwdHypot,FvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  fvar<var> x(3.0,1.3);
  double z(6.0);
  fvar<var> a = hypot(x,z);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * 6.0 * 6.0 / hypot(3.0,6.0) / (9.0 + 36.0),g[0]);
}
TEST(AgradFwdHypot,Double_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  double x(3.0);
  fvar<var> z(6.0,1.0);
  fvar<var> a = hypot(x,z);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0 * 3.0 * 3.0 / hypot(3.0,6.0) / (9.0 + 36.0),g[0]);
}


TEST(AgradFwdHypot,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = hypot(x,y);

  EXPECT_FLOAT_EQ(hypot(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(3.0 / hypot(3.0,6.0), a.val_.d_.val());
  EXPECT_FLOAT_EQ(6.0 / hypot(3.0,6.0), a.d_.val_.val());
  EXPECT_FLOAT_EQ(-0.059628479, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(3.0 / hypot(3.0,6.0), g[0]);
  EXPECT_FLOAT_EQ(6.0 / hypot(3.0,6.0), g[1]);
}
TEST(AgradFwdHypot,FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  double y(6.0);

  fvar<fvar<var> > a = hypot(x,y);

  EXPECT_FLOAT_EQ(hypot(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(3.0 / hypot(3.0,6.0), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(3.0 / hypot(3.0,6.0), g[0]);
}

TEST(AgradFwdHypot,Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  double x(3.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = hypot(x,y);

  EXPECT_FLOAT_EQ(hypot(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(6.0 / hypot(3.0,6.0), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(6.0 / hypot(3.0,6.0), g[0]);
}
TEST(AgradFwdHypot,FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = hypot(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);

  EXPECT_FLOAT_EQ(36.0 / hypot(3.0,6.0) / (9.0 + 36.0),g[0]);
  EXPECT_FLOAT_EQ(-2.0/15.0/std::sqrt(5.0), g[1]);
}
TEST(AgradFwdHypot,FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = hypot(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-2.0/15.0/std::sqrt(5.0), g[0]);
  EXPECT_FLOAT_EQ((3.0 * 3.0) / hypot(3.0,6.0) / (9.0 + 36.0),g[1]);
}
TEST(AgradFwdHypot,FvarFvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  double y(6.0);

  fvar<fvar<var> > a = hypot(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);

  EXPECT_FLOAT_EQ(6.0 * 6.0 / hypot(3.0,6.0) / (9.0 + 36.0),g[0]);
}

TEST(AgradFwdHypot,Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  double x(3.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = hypot(x,y);

  EXPECT_FLOAT_EQ(hypot(3.0,6.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(6.0 / hypot(3.0,6.0), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ((3.0 * 3.0) / hypot(3.0,6.0) / (9.0 + 36.0),g[0]);
}
TEST(AgradFwdHypot,FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = hypot(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.0079504643, g[0]);
  EXPECT_FLOAT_EQ(0.013913312,g[1]);
}
TEST(AgradFwdHypot,FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  fvar<fvar<var> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;
  double y(6.0);

  fvar<fvar<var> > a = hypot(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);

  EXPECT_FLOAT_EQ(-0.02385139175999775676169785246647,g[0]);
}

TEST(AgradFwdHypot,Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::hypot;

  double x(3.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = hypot(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.0119256958799988783808489262332,g[0]);
}

struct hypot_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return hypot(arg1,arg2);
  }
};

TEST(AgradFwdHypot, nan) {
  hypot_fun hypot_;
  test_nan_mix(hypot_,3.0,5.0,false);
}
