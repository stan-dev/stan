#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/falling_factorial.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/lgamma.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/fwd/scal/fun/trunc.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>

TEST(AgradFwdFallingFactorial,Fvar) {
  using stan::math::fvar;
  using stan::math::falling_factorial;
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

TEST(AgradFwdFallingFactorial,FvarFvarDouble) {
  using stan::math::fvar;
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
  test_nan_fwd(falling_factorial_,3.0,5.0,false);
}
