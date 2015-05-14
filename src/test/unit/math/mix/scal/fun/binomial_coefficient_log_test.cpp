#include <gtest/gtest.h>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>
#include <stan/math/fwd/scal/fun/binomial_coefficient_log.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/rev/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/lgamma.hpp>
#include <stan/math/rev/scal/fun/lgamma.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/rev/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/rev/scal/fun/tan.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>



TEST(AgradFwdBinomialCoefficientLog,FvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using boost::math::digamma;

  fvar<var> x(2004.0,1.0);
  double z(1002);
  fvar<var> a = binomial_coefficient_log(x,z);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val());
  EXPECT_FLOAT_EQ(0.69289774, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0.69289774, g[0]);
}
TEST(AgradFwdBinomialCoefficientLog,Double_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using boost::math::digamma;

  double x(2004.0);
  fvar<var> z(1002.0,2.0);
  fvar<var> a = binomial_coefficient_log(x,z);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val());
  EXPECT_NEAR(0, a.d_.val(),1e-8);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_NEAR(0, g[0],1e-8);
}
TEST(AgradFwdBinomialCoefficientLog,FvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using boost::math::digamma;

  fvar<var> x(2004.0,1.0);
  double z(1002);
  fvar<var> a = binomial_coefficient_log(x,z);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val());
  EXPECT_FLOAT_EQ(0.69289774, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.00049862865, g[0]);
}
TEST(AgradFwdBinomialCoefficientLog,Double_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using boost::math::digamma;

  double x(2004.0);
  fvar<var> z(1002.0,2.0);
  fvar<var> a = binomial_coefficient_log(x,z);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val());
  EXPECT_NEAR(0, a.d_.val(),1e-8);

  AVEC y = createAVEC(z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.00399002460681026, g[0]);
}


TEST(AgradFwdBinomialCoefficientLog,FvarVar_FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using boost::math::digamma;

  fvar<var> x(2004.0,1.0);
  fvar<var> z(1002.0,2.0);
  fvar<var> a = binomial_coefficient_log(x,z);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val());
  EXPECT_FLOAT_EQ(0.69289774, a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0.69289774, g[0]);
  EXPECT_NEAR(0, g[1],1e-8);
}
TEST(AgradFwdBinomialCoefficientLog,FvarVar_FvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using boost::math::digamma;

  fvar<var> x(2004.0,1.0);
  fvar<var> z(1002.0,2.0);
  fvar<var> a = binomial_coefficient_log(x,z);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val());
  EXPECT_FLOAT_EQ(0.69289774, a.d_.val());

  AVEC y = createAVEC(x.val_,z.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0.0014963837, g[0]);
  EXPECT_FLOAT_EQ(-0.0029925184551076781, g[1]);
}



TEST(AgradFwdBinomialCoefficientLog,FvarFvarVar_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using stan::math::binomial_coefficient_log;

  fvar<fvar<var> > x;
  x.val_.val_ = 2004.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1002.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = binomial_coefficient_log(x,y);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0.69289774, a.val_.d_.val());
  EXPECT_NEAR(0, a.d_.val_.val(),1e-8);
  EXPECT_FLOAT_EQ(0.0009975062, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.69289774, g[0]);
  EXPECT_NEAR(0, g[1],1e-8);
}
TEST(AgradFwdBinomialCoefficientLog,FvarFvarVar_Double_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using stan::math::binomial_coefficient_log;

  fvar<fvar<var> > x;
  x.val_.val_ = 2004.0;
  x.val_.d_ = 1.0;

  double y(1002.0);

  fvar<fvar<var> > a = binomial_coefficient_log(x,y);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0.69289774, a.val_.d_.val());
  EXPECT_NEAR(0, a.d_.val_.val(),1e-8);
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.69289774, g[0]);
}
TEST(AgradFwdBinomialCoefficientLog,Double_FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using stan::math::binomial_coefficient_log;

  double x(2004.0);

  fvar<fvar<var> > y;
  y.val_.val_ = 1002.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = binomial_coefficient_log(x,y);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_NEAR(0, a.d_.val_.val(),1e-8);
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_NEAR(0, g[0],1e-8);
}

TEST(AgradFwdBinomialCoefficientLog,FvarFvarVar_FvarFvarVar_2ndDeriv_x) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using stan::math::binomial_coefficient_log;

  fvar<fvar<var> > x;
  x.val_.val_ = 2004.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1002.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = binomial_coefficient_log(x,y);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0.69289774, a.val_.d_.val());
  EXPECT_NEAR(0, a.d_.val_.val(),1e-8);
  EXPECT_FLOAT_EQ(0.0009975062, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.00049862865, g[0]);
  EXPECT_FLOAT_EQ(0.00099750615170258105, g[1]);
}
TEST(AgradFwdBinomialCoefficientLog,FvarFvarVar_FvarFvarVar_2ndDeriv_y) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using stan::math::binomial_coefficient_log;

  fvar<fvar<var> > x;
  x.val_.val_ = 2004.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1002.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = binomial_coefficient_log(x,y);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0.69289774, a.val_.d_.val());
  EXPECT_NEAR(0, a.d_.val_.val(),1e-8);
  EXPECT_FLOAT_EQ(0.0009975062, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.0009975062, g[0]);
  EXPECT_FLOAT_EQ(-0.0019950123034051291, g[1]);
}
TEST(AgradFwdBinomialCoefficientLog,Double_FvarFvarVar_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using stan::math::binomial_coefficient_log;

  double x(2004.0);

  fvar<fvar<var> > y;
  y.val_.val_ = 1002.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = binomial_coefficient_log(x,y);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0, a.val_.d_.val());
  EXPECT_NEAR(0, a.d_.val_.val(),1e-8);
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.val_.grad(p,g);
  EXPECT_NEAR(-0.00199501230340513, g[0],1e-8);
}
TEST(AgradFwdBinomialCoefficientLog,FvarFvarVar_Double_2ndDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using stan::math::binomial_coefficient_log;

  fvar<fvar<var> > x;
  x.val_.val_ = 2004.0;
  x.val_.d_ = 1.0;

  double y(1002.0);

  fvar<fvar<var> > a = binomial_coefficient_log(x,y);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0.69289774, a.val_.d_.val());
  EXPECT_NEAR(0, a.d_.val_.val(),1e-8);
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.00049862863648177515, g[0]);
}
TEST(AgradFwdBinomialCoefficientLog,FvarFvarVar_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using stan::math::binomial_coefficient_log;

  fvar<fvar<var> > x;
  x.val_.val_ = 2004.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > y;
  y.val_.val_ = 1002.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > a = binomial_coefficient_log(x,y);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0.69289774, a.val_.d_.val());
  EXPECT_NEAR(0, a.d_.val_.val(),1e-8);
  EXPECT_FLOAT_EQ(0.0009975062, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_,y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-9.9501847e-07, g[0]);
  EXPECT_FLOAT_EQ(9.9501847e-07, g[1]);
}
TEST(AgradFwdBinomialCoefficientLog,Double_FvarFvarVar_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using stan::math::binomial_coefficient_log;

  double x(2004.0);

  fvar<fvar<var> > y;
  y.val_.val_ = 1002.0;
  y.d_.val_ = 1.0;
  y.val_.d_ = 1.0;

  fvar<fvar<var> > a = binomial_coefficient_log(x,y);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val_.val());
  EXPECT_NEAR(0, a.val_.d_.val(),1e-8);
  EXPECT_NEAR(0, a.d_.val_.val(),1e-8);
  EXPECT_FLOAT_EQ(-0.0019950124, a.d_.d_.val());

  AVEC p = createAVEC(y.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_NEAR(0, g[0],1e-8);
}
TEST(AgradFwdBinomialCoefficientLog,FvarFvarVar_Double_3rdDeriv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::binomial_coefficient_log;
  using stan::math::binomial_coefficient_log;

  fvar<fvar<var> > x;
  x.val_.val_ = 2004.0;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  double y(1002.0);

  fvar<fvar<var> > a = binomial_coefficient_log(x,y);

  EXPECT_FLOAT_EQ(binomial_coefficient_log(2004.0,1002.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0.69289774, a.val_.d_.val());
  EXPECT_FLOAT_EQ(0.69289774181268948, a.d_.val_.val());
  EXPECT_FLOAT_EQ(-0.00049862865, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(7.4613968e-07, g[0]);
}

struct binomial_coefficient_log_fun {
  template <typename T0, typename T1>
  inline 
  typename boost::math::tools::promote_args<T0,T1>::type
  operator()(const T0 arg1,
             const T1 arg2) const {
    return binomial_coefficient_log(arg1,arg2);
  }
};

TEST(AgradFwdBinomialCoefficientLog, nan) {
  binomial_coefficient_log_fun binomial_coefficient_log_;
  test_nan_mix(binomial_coefficient_log_,3.0,5.0,false);
}
