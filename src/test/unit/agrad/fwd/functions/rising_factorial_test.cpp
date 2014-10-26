#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdRisingFactorial, Fvar) {
  using stan::agrad::fvar;
  using stan::agrad::rising_factorial;
  using boost::math::digamma;

  fvar<double> a(4.0,1.0);
  fvar<double> x = rising_factorial(a,1);
  EXPECT_FLOAT_EQ(4.0, x.val_);
  EXPECT_FLOAT_EQ(1.0, x.d_);

  fvar<double> c(-3.0,2.0);

  EXPECT_THROW(rising_factorial(c, 2), std::domain_error);
  EXPECT_THROW(rising_factorial(2, c), std::domain_error);
  EXPECT_THROW(rising_factorial(c, c), std::domain_error);

  x = rising_factorial(a,a);
  EXPECT_FLOAT_EQ(840.0, x.val_);
  EXPECT_FLOAT_EQ(840.0 * (2 * digamma(8) - digamma(4)), x.d_);

  x = rising_factorial(5, a);
  EXPECT_FLOAT_EQ(1680.0, x.val_);
  EXPECT_FLOAT_EQ(1680.0 * digamma(9), x.d_);
}
TEST(AgradFwdRisingFactorial, FvarVar_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::rising_factorial;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::rising_factorial;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::rising_factorial;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::rising_factorial;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::rising_factorial;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::rising_factorial;
  using boost::math::digamma;

  double a(4.0);
  fvar<var> b(4.0,1.0);
  fvar<var> c = rising_factorial(a,b);

  AVEC y = createAVEC(b.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(3524.5959, g[0]);
}
TEST(AgradFwdRisingFactorial, FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::rising_factorial;
  using boost::math::digamma;

  fvar<fvar<double> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  fvar<fvar<double> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = rising_factorial(x,y);

  EXPECT_FLOAT_EQ((840.0), a.val_.val_);
  EXPECT_FLOAT_EQ(840. * (digamma(8) - digamma(4)), a.val_.d_);
  EXPECT_FLOAT_EQ(840 * digamma(8), a.d_.val_);
  EXPECT_FLOAT_EQ(1397.8143, a.d_.d_);
}
TEST(AgradFwdRisingFactorial, FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  test_nan(rising_factorial_,3.0,5.0,false);
}
