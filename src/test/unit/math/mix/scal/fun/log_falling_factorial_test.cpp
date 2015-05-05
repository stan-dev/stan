#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/log_falling_factorial.hpp>
#include <stan/math/rev/scal/fun/log_falling_factorial.hpp>
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


TEST(AgradFwdLogFallingFactorial,FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  fvar<var> a(4.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c = log_falling_factorial(a,b);

  EXPECT_FLOAT_EQ(log(4), c.val_.val());
  EXPECT_FLOAT_EQ(0.25, c.d_.val());

  AVEC y = createAVEC(a.val_,b.val_);
  VEC g;
  c.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.5061177, g[0]);
  EXPECT_FLOAT_EQ(-1.2561177, g[1]);
}
TEST(AgradFwdLogFallingFactorial,FvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  fvar<var> a(4.0,1.0);
  double b(3.0); 
  fvar<var> c = log_falling_factorial(a,b);

  EXPECT_FLOAT_EQ(log(4), c.val_.val());
  EXPECT_FLOAT_EQ(1.5061177, c.d_.val());

  AVEC y = createAVEC(a.val_);
  VEC g;
  c.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.5061177, g[0]);
}
TEST(AgradFwdLogFallingFactorial,Double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  double a(4.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c = log_falling_factorial(a,b);

  EXPECT_FLOAT_EQ(log(4), c.val_.val());
  EXPECT_FLOAT_EQ(-1.2561177, c.d_.val());

  AVEC y = createAVEC(b.val_);
  VEC g;
  c.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-1.2561177, g[0]);
}
TEST(AgradFwdLogFallingFactorial,FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  fvar<var> a(4.0,1.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c = log_falling_factorial(a,b);

  AVEC y = createAVEC(a.val_,b.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.22132295, g[0]);
  EXPECT_FLOAT_EQ(-0.28382295, g[1]);
}
TEST(AgradFwdLogFallingFactorial,FvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  fvar<var> a(4.0,1.0);
  double b(3.0); 
  fvar<var> c = log_falling_factorial(a,b);

  AVEC y = createAVEC(a.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.22132295, g[0]);
}
TEST(AgradFwdLogFallingFactorial,Double_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  double a(4.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c = log_falling_factorial(a,b);

  AVEC y = createAVEC(b.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.28382295, g[0]);
}
TEST(AgradFwdLogFallingFactorial,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_falling_factorial(x,y);

  EXPECT_FLOAT_EQ(1.3862944, a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5061177, a.val_.d_.val());
  EXPECT_FLOAT_EQ(-1.2561177, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.5061177, g[0]);
  EXPECT_FLOAT_EQ(-1.2561177, g[1]);
}
TEST(AgradFwdLogFallingFactorial,FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  double y(3.0);
  fvar<fvar<var> > a = log_falling_factorial(x,y);

  EXPECT_FLOAT_EQ(1.3862944, a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5061177, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.5061177, g[0]);
}


TEST(AgradFwdLogFallingFactorial,Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  double x(4.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_falling_factorial(x,y);

  EXPECT_FLOAT_EQ(1.3862944, a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(-1.2561177, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-1.2561177, g[0]);
}
TEST(AgradFwdLogFallingFactorial,FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_falling_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.22132295, g[0]);
  EXPECT_FLOAT_EQ(0, g[1]);
}
TEST(AgradFwdLogFallingFactorial,FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_falling_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(-0.28382295, g[1]);
}
TEST(AgradFwdLogFallingFactorial,FvarFvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  double y(3.0);
  fvar<fvar<var> > a = log_falling_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.22132295, g[0]);
}
TEST(AgradFwdLogFallingFactorial,Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  double x(4.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_falling_factorial(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.28382295, g[0]);
}
TEST(AgradFwdLogFallingFactorial,FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_falling_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0, g[0]);
  EXPECT_FLOAT_EQ(0, g[1]);
}
TEST(AgradFwdLogFallingFactorial,FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  double y(3.0);
  fvar<fvar<var> > a = log_falling_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.048789728, g[0]);
}
TEST(AgradFwdLogFallingFactorial,Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_falling_factorial;

  double x(4.0);
  fvar<fvar<var> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = log_falling_factorial(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.080039732, g[0]);
}

struct log_falling_factorial_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return log_falling_factorial(arg1,arg2);
  }
};

TEST(AgradFwdLogFallingFactorial, nan) {
  log_falling_factorial_fun log_falling_factorial_;
  test_nan_mix(log_falling_factorial_,3.0,5.0,false);
}
