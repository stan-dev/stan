#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/digamma.hpp>
#include <stan/math/rev/scal/fun/digamma.hpp>
#include <stan/math/fwd/scal/fun/rising_factorial.hpp>
#include <stan/math/rev/scal/fun/rising_factorial.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/rev/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/rev/scal/fun/tan.hpp>
#include <stan/math/prim/scal/fun/trigamma.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>

TEST(AgradFwdRisingFactorial, FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  fvar<var> a(4.0,1.0);
  fvar<var> b(4.0,1.0);
  fvar<var> c = rising_factorial(a,b);

  EXPECT_FLOAT_EQ((840.0), c.val_.val());
  EXPECT_FLOAT_EQ(840. * (2 * digamma(8) - digamma(4)), c.d_.val());

  AVEC y = createAVEC(a.val_,b.val_);
  VEC g;
  c.val_.grad(y,g);
  EXPECT_FLOAT_EQ(840. * (digamma(8) - digamma(4)), g[0]);
  EXPECT_FLOAT_EQ(840 * digamma(8), g[1]);
}
TEST(AgradFwdRisingFactorial, FvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  fvar<var> a(4.0,1.0);
  double b(4.0);
  fvar<var> c = rising_factorial(a,b);

  EXPECT_FLOAT_EQ((840.0), c.val_.val());
  EXPECT_FLOAT_EQ(840. * (digamma(8) - digamma(4)), c.d_.val());

  AVEC y = createAVEC(a.val_);
  VEC g;
  c.val_.grad(y,g);
  EXPECT_FLOAT_EQ(840. * (digamma(8) - digamma(4)), g[0]);
}
TEST(AgradFwdRisingFactorial, Double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  double a(4.0);
  fvar<var> b(4.0,1.0);
  fvar<var> c = rising_factorial(a,b);

  EXPECT_FLOAT_EQ((840.0), c.val_.val());
  EXPECT_FLOAT_EQ(840 * digamma(8), c.d_.val());

  AVEC y = createAVEC(b.val_);
  VEC g;
  c.val_.grad(y,g);
  EXPECT_FLOAT_EQ(840 * digamma(8), g[0]);
}
TEST(AgradFwdRisingFactorial, FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  fvar<var> a(4.0,1.0);
  fvar<var> b(4.0,1.0);
  fvar<var> c = rising_factorial(a,b);

  AVEC y = createAVEC(a.val_,b.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1755.8143, g[0]);
  EXPECT_FLOAT_EQ(4922.4102, g[1]);
}
TEST(AgradFwdRisingFactorial, FvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  fvar<var> a(4.0,1.0);
  double b(4.0);
  fvar<var> c = rising_factorial(a,b);

  AVEC y = createAVEC(a.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(358, g[0]);
}
TEST(AgradFwdRisingFactorial, Double_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  double a(4.0);
  fvar<var> b(4.0,1.0);
  fvar<var> c = rising_factorial(a,b);

  AVEC y = createAVEC(b.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(3524.5959, g[0]);
}

TEST(AgradFwdRisingFactorial, FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = rising_factorial(x,y);

  EXPECT_FLOAT_EQ((840.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(840. * (digamma(8) - digamma(4)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(840 * digamma(8), a.d_.val_.val());
  EXPECT_FLOAT_EQ(1397.8143, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(840. * (digamma(8) - digamma(4)),g[0]);
  EXPECT_FLOAT_EQ(840 * digamma(8), g[1]);
}
TEST(AgradFwdRisingFactorial, FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  double y(4.0);

  fvar<fvar<var> > a = rising_factorial(x,y);

  EXPECT_FLOAT_EQ((840.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(840. * (digamma(8) - digamma(4)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(840. * (digamma(8) - digamma(4)),g[0]);
}
TEST(AgradFwdRisingFactorial, Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  double x(4.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = rising_factorial(x,y);

  EXPECT_FLOAT_EQ((840.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(840 * digamma(8), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(840 * digamma(8), g[0]);
}
TEST(AgradFwdRisingFactorial, FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = rising_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(358,g[0]);
  EXPECT_FLOAT_EQ(1397.8143, g[1]);
}
TEST(AgradFwdRisingFactorial, FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = rising_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1397.8143, g[0]);
  EXPECT_FLOAT_EQ(3524.5959,g[1]);
}
TEST(AgradFwdRisingFactorial, FvarFvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  double y(4.0);

  fvar<fvar<var> > a = rising_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(358,g[0]);
}
TEST(AgradFwdRisingFactorial, Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  double x(4.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = rising_factorial(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(3524.5959, g[0]);
}
TEST(AgradFwdRisingFactorial, FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = rising_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(876.61487, g[0]);
  EXPECT_FLOAT_EQ(3112.9858,g[1]);
}
TEST(AgradFwdRisingFactorial, FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;
  double y(4.0);

  fvar<fvar<var> > a = rising_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(132,g[0]);
}
TEST(AgradFwdRisingFactorial, Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  double x(4.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = rising_factorial(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(7540.293, g[0]);
}

struct rising_factorial_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return rising_factorial(arg1,arg2);
  }
};

TEST(AgradFwdRisingFactorial, nan) {
  rising_factorial_fun rising_factorial_;
  test_nan_mix(rising_factorial_,3.0,5.0,false);
}

