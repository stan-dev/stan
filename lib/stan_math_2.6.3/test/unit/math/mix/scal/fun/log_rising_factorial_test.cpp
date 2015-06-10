#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/fwd/scal/fun/log_rising_factorial.hpp>
#include <stan/math/rev/scal/fun/log_rising_factorial.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/rev/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/rev/scal/fun/tan.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>

TEST(AgradFwdLogRisingFactorial,FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<var> a(4.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c = log_rising_factorial(a,b);

  EXPECT_FLOAT_EQ(std::log(120.0), c.val_.val());
  EXPECT_FLOAT_EQ(2.4894509, c.d_.val());

  AVEC y = createAVEC(a.val_,b.val_);
  VEC g;
  c.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0.61666667, g[0]);
  EXPECT_FLOAT_EQ(1.8727844, g[1]);
}
TEST(AgradFwdLogRisingFactorial,FvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<var> a(4.0,1.0);
  double b(3.0);
  fvar<var> c = log_rising_factorial(a,b);

  EXPECT_FLOAT_EQ(std::log(120.0), c.val_.val());
  EXPECT_FLOAT_EQ(0.61666667, c.d_.val());

  AVEC y = createAVEC(a.val_);
  VEC g;
  c.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0.61666667, g[0]);
}
TEST(AgradFwdLogRisingFactorial,Double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  double a(4.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c = log_rising_factorial(a,b);

  EXPECT_FLOAT_EQ(std::log(120.0), c.val_.val());
  EXPECT_FLOAT_EQ(1.8727844, c.d_.val());

  AVEC y = createAVEC(b.val_);
  VEC g;
  c.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.8727844, g[0]);
}

TEST(AgradFwdLogRisingFactorial,FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<var> a(4.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c = log_rising_factorial(a,b);

  AVEC y = createAVEC(a.val_,b.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.023267401, g[0]);
  EXPECT_FLOAT_EQ(0.30709034, g[1]);
}
TEST(AgradFwdLogRisingFactorial,FvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<var> a(4.0,1.0);
  double b(3.0);
  fvar<var> c = log_rising_factorial(a,b);

  AVEC y = createAVEC(a.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.13027778, g[0]);
}
TEST(AgradFwdLogRisingFactorial,Double_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  double a(4.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c = log_rising_factorial(a,b);

  AVEC y = createAVEC(b.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.15354517, g[0]);
}
TEST(AgradFwdLogRisingFactorial,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  EXPECT_FLOAT_EQ(std::log(120.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0.61666667, a.val_.d_.val());
  EXPECT_FLOAT_EQ(1.8727844, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0.15354517, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.61666667, g[0]);
  EXPECT_FLOAT_EQ(1.8727844, g[1]);
}
TEST(AgradFwdLogRisingFactorial,FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  double y(3.0);

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  EXPECT_FLOAT_EQ(std::log(120.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0.61666667, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.61666667, g[0]);
}
TEST(AgradFwdLogRisingFactorial,Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  double x(4.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  EXPECT_FLOAT_EQ(std::log(120.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(1.8727844, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.8727844, g[0]);
}
TEST(AgradFwdLogRisingFactorial,FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.13027778, g[0]);
  EXPECT_FLOAT_EQ(0.15354517, g[1]);
}
TEST(AgradFwdLogRisingFactorial,FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.15354517, g[0]);
  EXPECT_FLOAT_EQ(0.15354517, g[1]);
}
TEST(AgradFwdLogRisingFactorial,FvarFvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  double y(3.0);

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.13027778, g[0]);
}
TEST(AgradFwdLogRisingFactorial,Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  double x(4.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.15354517, g[0]);
}
TEST(AgradFwdLogRisingFactorial,FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.023530472, g[0]);
  EXPECT_FLOAT_EQ(-0.023530472, g[1]);
}
TEST(AgradFwdLogRisingFactorial,FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;
  double y(3.0);

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.056509256, g[0]);
}
TEST(AgradFwdLogRisingFactorial,Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  double x(4.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = log_rising_factorial(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.023530472, g[0]);
}

struct log_rising_factorial_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return log_rising_factorial(arg1,arg2);
  }
};

TEST(AgradFwdLogRisingFactorial, nan) {
  log_rising_factorial_fun log_rising_factorial_;
  test_nan_mix(log_rising_factorial_,3.0,5.0,false);
}
