#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/log_falling_factorial.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/lgamma.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>

TEST(AgradFwdLogFallingFactorial,Fvar) {
  using stan::math::fvar;
  using stan::math::log_falling_factorial;
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

TEST(AgradFwdLogFallingFactorial,FvarFvarDouble) {
  using stan::math::fvar;
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
  test_nan_fwd(log_falling_factorial_,3.0,5.0,false);
}
