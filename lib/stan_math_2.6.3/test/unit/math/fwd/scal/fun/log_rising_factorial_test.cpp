#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/log_rising_factorial.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/digamma.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/lgamma.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>

TEST(AgradFwdLogRisingFactorial,Fvar) {
  using stan::math::fvar;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<double> a(4.0,1.0);
  fvar<double> x = log_rising_factorial(a,1.0);
  EXPECT_FLOAT_EQ(std::log(4.0), x.val_);
  EXPECT_FLOAT_EQ(0.25, x.d_);

  fvar<double> c(-3.0,2.0);

  EXPECT_THROW(log_rising_factorial(c, 2), std::domain_error);
  EXPECT_THROW(log_rising_factorial(2, c), std::domain_error);
  EXPECT_THROW(log_rising_factorial(c, c), std::domain_error);

  x = log_rising_factorial(a,a);
  EXPECT_FLOAT_EQ(std::log(840.0), x.val_);
  EXPECT_FLOAT_EQ((2 * digamma(8) - digamma(4)), x.d_);

  x = log_rising_factorial(5, a);
  EXPECT_FLOAT_EQ(std::log(1680.0), x.val_);
  EXPECT_FLOAT_EQ(digamma(9), x.d_);
}

TEST(AgradFwdLogRisingFactorial,FvarFvarDouble) {
  using stan::math::fvar;
  using stan::math::log_rising_factorial;
  using boost::math::digamma;

  fvar<fvar<double> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 3.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = log_rising_factorial(x,y);

  EXPECT_FLOAT_EQ(std::log(120.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0.61666667, a.val_.d_);
  EXPECT_FLOAT_EQ(1.8727844, a.d_.val_);
  EXPECT_FLOAT_EQ(0.15354517, a.d_.d_);
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
  test_nan_fwd(log_rising_factorial_,3.0,5.0,false);
}
