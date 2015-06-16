#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <stan/math/fwd/scal/fun/lbeta.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/digamma.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/lgamma.hpp>
#include <stan/math/fwd/scal/fun/lmgamma.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>

TEST(AgradFwdLbeta,Fvar) {
  using stan::math::fvar;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<double> x(0.5,1.0);
  fvar<double> y(1.2,2.0);

  double w = 1.3;

  fvar<double> a = lbeta(x, y);
  EXPECT_FLOAT_EQ(lbeta(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ(digamma(0.5) + 2.0 * digamma(1.2) 
                  - (1.0 + 2.0) * digamma(0.5 + 1.2), a.d_);

  fvar<double> b = lbeta(x, w);
  EXPECT_FLOAT_EQ(lbeta(0.5, 1.3), b.val_);
  EXPECT_FLOAT_EQ(1.0 * digamma(0.5) - 1.0 * digamma(0.5 + 1.3), b.d_);

  fvar<double> c = lbeta(w, x);
  EXPECT_FLOAT_EQ(lbeta(1.3, 0.5), c.val_);
  EXPECT_FLOAT_EQ(1.0 * digamma(0.5) - 1.0 * digamma(1.3 + 0.5), c.d_);
}

TEST(AgradFwdLbeta,FvarFvarDouble) {
  using stan::math::fvar;
  using boost::math::digamma;
  using stan::math::lbeta;

  fvar<fvar<double> > x;
  x.val_.val_ = 3.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 6.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = lbeta(x,y);

  EXPECT_FLOAT_EQ(lbeta(3.0,6.0), a.val_.val_);
  EXPECT_FLOAT_EQ(digamma(3.0) - digamma(9.0), a.val_.d_);
  EXPECT_FLOAT_EQ(digamma(6.0) - digamma(9.0), a.d_.val_);
  EXPECT_FLOAT_EQ(-0.11751202, a.d_.d_);
}

struct lbeta_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return lbeta(arg1,arg2);
  }
};

TEST(AgradFwdLbeta, nan) {
  lbeta_fun lbeta_;
  test_nan_fwd(lbeta_,3.0,5.0,false);
}
