#include <gtest/gtest.h>
#include <boost/math/special_functions/erf.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/erf.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>

TEST(AgradFwdErf,Fvar) {
  using stan::math::fvar;
  using std::exp;
  using std::sqrt;
  using boost::math::erf;

  fvar<double> x(0.5,1.0);

  fvar<double> a = erf(x);
  EXPECT_FLOAT_EQ(erf(0.5), a.val_);
  EXPECT_FLOAT_EQ(2 * exp(-0.5 * 0.5) / 
                  sqrt(boost::math::constants::pi<double>()), a.d_);

 fvar<double> b = erf(-x);
  EXPECT_FLOAT_EQ(erf(-0.5), b.val_);
  EXPECT_FLOAT_EQ(-2 * exp(-0.5 * 0.5) 
                  / sqrt(boost::math::constants::pi<double>()), b.d_);
}

TEST(AgradFwdErf,FvarFvarDouble) {
  using stan::math::fvar;
  using std::exp;
  using std::sqrt;
  using boost::math::erf;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = erf(x);

  EXPECT_FLOAT_EQ(erf(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(2 * exp(-0.5 * 0.5) / 
                  sqrt(boost::math::constants::pi<double>()), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = erf(y);
  EXPECT_FLOAT_EQ(erf(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(2 * exp(-0.5 * 0.5) / 
                  sqrt(boost::math::constants::pi<double>()), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}


struct erf_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return erf(arg1);
  }
};

TEST(AgradFwdErf,erf_NaN) {
  erf_fun erf_;
  test_nan_fwd(erf_,false);
}
