#include <gtest/gtest.h>
#include <boost/math/special_functions/gamma.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/gamma_q.hpp>
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

TEST(AgradFwdGammaQ, Fvar){
  using stan::math::fvar;
  using stan::math::gamma_q;
  using boost::math::gamma_q;

  fvar<double> x(0.5);
  x.d_ = 1.0;
  fvar<double> y (1.0);
  y.d_ = 1.0;

  fvar<double> a = gamma_q(x,y);
  EXPECT_FLOAT_EQ(gamma_q(0.5,1.0), a.val_);
  EXPECT_FLOAT_EQ(0.18228334, a.d_);

  double z = 1.0;
  double w = 0.5;

  a = gamma_q(x,z);
  EXPECT_FLOAT_EQ(gamma_q(0.5,1.0), a.val_);
  EXPECT_FLOAT_EQ(0.38983709, a.d_);

  a = gamma_q(w,y);
  EXPECT_FLOAT_EQ(gamma_q(0.5,1.0), a.val_);
  EXPECT_FLOAT_EQ(-boost::math::gamma_p_derivative(0.5,1.0), a.d_);

  EXPECT_THROW(gamma_q(-x,y), std::domain_error);
  EXPECT_THROW(gamma_q(x,-y), std::domain_error);
}

TEST(AgradFwdGammaQ, FvarFvarDouble) {
  using stan::math::fvar;
  using boost::math::gamma_q;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<double> > a = gamma_q(x,y);

  EXPECT_FLOAT_EQ(gamma_q(0.5,1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0.38983709, a.val_.d_);
  EXPECT_FLOAT_EQ(-boost::math::gamma_p_derivative(0.5,1.0), a.d_.val_);
  EXPECT_FLOAT_EQ(-0.40753385, a.d_.d_);
}

struct gamma_q_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return gamma_q(arg1,arg2);
  }
};

TEST(AgradFwdGammaQ, nan) {
  gamma_q_fun gamma_q_;
  test_nan_fwd(gamma_q_,3.0,5.0,false);
}
