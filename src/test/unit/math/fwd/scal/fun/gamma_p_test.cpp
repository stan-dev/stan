#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/gamma_p.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/ceil.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/fwd/scal/fun/tgamma.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>

TEST(AgradFwdGammaP, gamma_p){
  using stan::math::fvar;
  using stan::math::gamma_p;
  using boost::math::gamma_p;

  fvar<double> x(0.5);
  x.d_ = 1.0;
  fvar<double> y (1.0);
  y.d_ = 1.0;

  fvar<double> a = gamma_p(x,y);
  EXPECT_FLOAT_EQ(gamma_p(0.5,1.0), a.val_);
  EXPECT_FLOAT_EQ(-0.18228334, a.d_);

  double z = 1.0;
  double w = 0.5;

  a = gamma_p(x,z);
  EXPECT_FLOAT_EQ(gamma_p(0.5,1.0), a.val_);
  EXPECT_FLOAT_EQ(-0.389837, a.d_);

  a = gamma_p(w,y);
  EXPECT_FLOAT_EQ(gamma_p(0.5,1.0), a.val_);
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), a.d_);

  EXPECT_THROW(gamma_p(-x,y), std::domain_error);
  EXPECT_THROW(gamma_p(x,-y), std::domain_error);
}

TEST(AgradFwdGammaP, FvarFvarDouble) {
  using stan::math::fvar;
  using boost::math::gamma_p;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = gamma_p(x,y);

  EXPECT_FLOAT_EQ(gamma_p(0.5,1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(-0.38983709, a.val_.d_);
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), a.d_.val_);
  EXPECT_FLOAT_EQ(0.40753385, a.d_.d_);
}


struct gamma_p_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return gamma_p(arg1,arg2);
  }
};

TEST(AgradFwdGammaP, nan) {
  gamma_p_fun gamma_p_;
  test_nan_fwd(gamma_p_,3.0,5.0,false);
}
