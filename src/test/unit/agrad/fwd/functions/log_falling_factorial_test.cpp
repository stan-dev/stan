#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdLogFallingFactorial,Fvar) {
  using stan::agrad::fvar;
  using stan::agrad::log_falling_factorial;
  using boost::math::digamma;

  fvar<double> a(4.0,1.0);
  fvar<double> x = log_falling_factorial(a,1);
  EXPECT_FLOAT_EQ(std::log(24.0), x.val_);
  EXPECT_FLOAT_EQ(digamma(5), x.d_);

  fvar<double> c(-3.0,2.0);

  EXPECT_THROW(log_falling_factorial(c, 2), std::domain_error);
  EXPECT_THROW(log_falling_factorial(2, c), std::domain_error);
  EXPECT_THROW(log_falling_factorial(c, c), std::domain_error);

  x = log_falling_factorial(a,a);
  EXPECT_FLOAT_EQ(0.0, x.val_);
  EXPECT_FLOAT_EQ(0.0, x.d_);

  x = log_falling_factorial(5, a);
  EXPECT_FLOAT_EQ(std::log(5.0), x.val_);
  EXPECT_FLOAT_EQ(-digamma(5.0),x.d_);
}
TEST(AgradFwdLogFallingFactorial,FvarVar_FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_falling_factorial;

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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_falling_factorial;

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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_falling_factorial;

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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_falling_factorial;

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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_falling_factorial;

  fvar<var> a(4.0,1.0);
  double b(3.0); 
  fvar<var> c = log_falling_factorial(a,b);

  AVEC y = createAVEC(a.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.22132295, g[0]);
}
TEST(AgradFwdLogFallingFactorial,Double_FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::log_falling_factorial;

  double a(4.0);
  fvar<var> b(3.0,1.0);
  fvar<var> c = log_falling_factorial(a,b);

  AVEC y = createAVEC(b.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.28382295, g[0]);
}
TEST(AgradFwdLogFallingFactorial,FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::log_falling_factorial;

  fvar<fvar<double> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = log_falling_factorial(x,y);

  EXPECT_FLOAT_EQ(1.3862944, a.val_.val_);
  EXPECT_FLOAT_EQ(1.5061177, a.val_.d_);
  EXPECT_FLOAT_EQ(-1.2561177, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdLogFallingFactorial,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  using stan::agrad::fvar;
  using stan::agrad::var;
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
  test_nan(log_falling_factorial_,3.0,5.0,false);
}
