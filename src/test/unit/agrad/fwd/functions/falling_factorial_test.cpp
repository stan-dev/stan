#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdFallingFactorial,Fvar) {
  using stan::agrad::fvar;
  using stan::agrad::falling_factorial;
  using boost::math::digamma;

  fvar<double> a(4.0,1.0);
  fvar<double> x = falling_factorial(a,1);
  EXPECT_FLOAT_EQ(24.0, x.val_);
  EXPECT_FLOAT_EQ(24.0 * digamma(5), x.d_);

  fvar<double> c(-3.0,2.0);

  EXPECT_THROW(falling_factorial(c, 2), std::domain_error);
  EXPECT_THROW(falling_factorial(2, c), std::domain_error);
  EXPECT_THROW(falling_factorial(c, c), std::domain_error);

  x = falling_factorial(a,a);
  EXPECT_FLOAT_EQ(1.0, x.val_);
  EXPECT_FLOAT_EQ(0.0, x.d_);

  x = falling_factorial(5, a);
  EXPECT_FLOAT_EQ(5.0, x.val_);
  EXPECT_FLOAT_EQ(-5.0 * digamma(5.0),x.d_);
}
TEST(AgradFwdFallingFactorial,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::falling_factorial;

  fvar<var> a(4.0,1.0);
  fvar<var> b(4.0,1.0);
  fvar<var> c = falling_factorial(a,b);

  EXPECT_FLOAT_EQ(1, c.val_.val());
  EXPECT_FLOAT_EQ(0, c.d_.val());

  AVEC y = createAVEC(a.val_,b.val_);
  VEC g;
  c.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.5061177, g[0]);
  EXPECT_FLOAT_EQ(-1.5061177, g[1]);
}
TEST(AgradFwdFallingFactorial,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::falling_factorial;

  fvar<var> a(4.0,1.0);
  fvar<var> b(4.0,1.0);
  fvar<var> c = falling_factorial(a,b);

  AVEC y = createAVEC(a.val_,b.val_);
  VEC g;
  c.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.22132295, g[0]);
  EXPECT_FLOAT_EQ(-0.22132295, g[1]);
}
TEST(AgradFwdFallingFactorial,FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::falling_factorial;

  fvar<fvar<double> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = falling_factorial(x,y);

  EXPECT_FLOAT_EQ(falling_factorial(4,4.0), a.val_.val_);
  EXPECT_FLOAT_EQ(1.5061177, a.val_.d_);
  EXPECT_FLOAT_EQ(-1.5061177, a.d_.val_);
  EXPECT_FLOAT_EQ(-2.2683904, a.d_.d_);
}
TEST(AgradFwdFallingFactorial,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = falling_factorial(x,y);

  EXPECT_FLOAT_EQ(falling_factorial(4,4.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(1.5061177, a.val_.d_.val());
  EXPECT_FLOAT_EQ(-1.5061177, a.d_.val_.val());
  EXPECT_FLOAT_EQ(-2.2683904, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(1.5061177, g[0]);
  EXPECT_FLOAT_EQ(-1.5061177, g[1]);
}
TEST(AgradFwdFallingFactorial,FvarFvarVar_2ndDeriv_x) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = falling_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(2.4897134, g[0]);
  EXPECT_FLOAT_EQ(-2.2683904, g[1]);
}
TEST(AgradFwdFallingFactorial,FvarFvarVar_2ndDeriv_y) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = falling_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-2.2683904, g[0]);
  EXPECT_FLOAT_EQ(2.0470674, g[1]);
}
TEST(AgradFwdFallingFactorial,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::falling_factorial;

  fvar<fvar<var> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 4.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = falling_factorial(x,y);

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-3.7498014, g[0]);
  EXPECT_FLOAT_EQ(3.0831244, g[1]);
}

struct falling_factorial_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return falling_factorial(arg1,arg2);
  }
};

TEST(AgradFwdFallingFactorial, nan) {
  falling_factorial_fun falling_factorial_;
  test_nan(falling_factorial_,3.0,5.0,false);
}
