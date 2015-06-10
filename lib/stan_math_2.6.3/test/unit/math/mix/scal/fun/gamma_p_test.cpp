#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <stan/math/rev/scal/fun/gamma_p.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/gamma_p.hpp>
#include <stan/math/rev/scal/fun/gamma_p.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/rev/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/rev/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/ceil.hpp>
#include <stan/math/rev/scal/fun/ceil.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/rev/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/rev/scal/fun/tan.hpp>
#include <stan/math/fwd/scal/fun/tgamma.hpp>
#include <stan/math/rev/scal/fun/tgamma.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>


TEST(AgradFwdGammaP, FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(0.5,1.0);
  fvar<var> z(1.0,1.0);
  fvar<var> a = stan::math::gamma_p(x,z);

  EXPECT_FLOAT_EQ(boost::math::gamma_p(0.5,1.0), a.val_.val());
  EXPECT_FLOAT_EQ(-0.18228334, a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.38983709,g[0]);
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0),g[1]);
}
TEST(AgradFwdGammaP, Double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(0.5);
  fvar<var> z(1.0,1.0);
  fvar<var> a = stan::math::gamma_p(x,z);

  EXPECT_FLOAT_EQ(boost::math::gamma_p(0.5,1.0), a.val_.val());
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0),g[0]);
}
TEST(AgradFwdGammaP, FvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(0.5,1.0);
  double z(1.0);
  fvar<var> a = stan::math::gamma_p(x,z);

  EXPECT_FLOAT_EQ(boost::math::gamma_p(0.5,1.0), a.val_.val());
  EXPECT_FLOAT_EQ(-0.38983709, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.38983709,g[0]);
}
TEST(AgradFwdGammaP, FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(0.5,1.0);
  fvar<var> z(1.0,1.0);
  fvar<var> a = stan::math::gamma_p(x,z);

  EXPECT_FLOAT_EQ(boost::math::gamma_p(0.5,1.0), a.val_.val());
  EXPECT_FLOAT_EQ(-0.18228334, a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.19403456,g[0]);
  EXPECT_FLOAT_EQ(0.096204743,g[1]);
}
TEST(AgradFwdGammaP, Double_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  double x(0.5);
  fvar<var> z(1.0,1.0);
  fvar<var> a = stan::math::gamma_p(x,z);

  EXPECT_FLOAT_EQ(boost::math::gamma_p(0.5,1.0), a.val_.val());
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), a.d_.val());

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.31133062,g[0]);
}
TEST(AgradFwdGammaP, FvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> x(0.5,1.0);
  double z(1.0);
  fvar<var> a = stan::math::gamma_p(x,z);

  EXPECT_FLOAT_EQ(boost::math::gamma_p(0.5,1.0), a.val_.val());
  EXPECT_FLOAT_EQ(-0.38983709, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.21349931,g[0]);
}



TEST(AgradFwdGammaP, FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::gamma_p;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = gamma_p(x,y);

  EXPECT_FLOAT_EQ(gamma_p(0.5,1.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-0.38983709, a.val_.d_.val());
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0.40753385, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.38983709, g[0]);
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), g[1]);
}
TEST(AgradFwdGammaP, Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::gamma_p;

  double x(0.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = gamma_p(x,y);

  EXPECT_FLOAT_EQ(gamma_p(0.5,1.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), g[0]);
}
TEST(AgradFwdGammaP, FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::gamma_p;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  double y(1.0);

  fvar<fvar<var> > a = gamma_p(x,y);

  EXPECT_FLOAT_EQ(gamma_p(0.5,1.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-0.38983709, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.38983709, g[0]);
}

TEST(AgradFwdGammaP, FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::gamma_p;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = gamma_p(x,y);

  EXPECT_FLOAT_EQ(gamma_p(0.5,1.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-0.38983709, a.val_.d_.val());
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0.40753385, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.21349931, g[0]);
  EXPECT_FLOAT_EQ(0.40753537, g[1]);
}
TEST(AgradFwdGammaP, FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::gamma_p;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = gamma_p(x,y);

  EXPECT_FLOAT_EQ(gamma_p(0.5,1.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-0.38983709, a.val_.d_.val());
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0.40753385, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.40753385, g[0]);
  EXPECT_FLOAT_EQ(-0.31133062, g[1]);
}
TEST(AgradFwdGammaP, Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::gamma_p;

  double x(0.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = gamma_p(x,y);

  EXPECT_FLOAT_EQ(gamma_p(0.5,1.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.31133062, g[0]);
}
TEST(AgradFwdGammaP, FvarFvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::gamma_p;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  double y(1.0);

  fvar<fvar<var> > a = gamma_p(x,y);

  EXPECT_FLOAT_EQ(gamma_p(0.5,1.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-0.38983709, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.21349931, g[0]);
}

TEST(AgradFwdGammaP, FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::gamma_p;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = gamma_p(x,y);

  EXPECT_FLOAT_EQ(gamma_p(0.5,1.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-0.38983709, a.val_.d_.val());
  EXPECT_FLOAT_EQ(boost::math::gamma_p_derivative(0.5,1.0), a.d_.val_.val());
  EXPECT_FLOAT_EQ(0.40753385, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.22403987, g[0]);
  EXPECT_FLOAT_EQ(-0.40374705, g[1]);
}
TEST(AgradFwdGammaP, Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::gamma_p;

  double x(0.5);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = gamma_p(x,y);

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.57077283, g[0]);
}
TEST(AgradFwdGammaP, FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using boost::math::gamma_p;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  double y(1.0);

  fvar<fvar<var> > a = gamma_p(x,y);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.5462361, g[0]);
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
  test_nan_mix(gamma_p_,3.0,5.0,false);
}
