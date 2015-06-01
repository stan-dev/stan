#include <gtest/gtest.h>
#include <stan/math/fwd/scal/fun/inc_beta.hpp>
#include <stan/math/rev/scal/fun/inc_beta.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_beta.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/digamma.hpp>
#include <stan/math/rev/scal/fun/digamma.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/acos.hpp>
#include <stan/math/rev/scal/fun/acos.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/ceil.hpp>
#include <stan/math/rev/scal/fun/ceil.hpp>
#include <stan/math/fwd/scal/fun/lbeta.hpp>
#include <stan/math/fwd/scal/fun/grad_inc_beta.hpp>
#include <stan/math/rev/scal/fun/grad_inc_beta.hpp>

#include <stan/math/rev/mat/fun/Eigen_NumTraits.hpp>


TEST(ProbInternalMath, grad_reg_inc_beta_fd) {
  using stan::math::fvar;
  using stan::math::digamma;
  using stan::math::exp;
  using stan::math::lbeta;

  fvar<double> a = 1.0;
  fvar<double> b = 1.0;
  fvar<double> g = 0.4;
  a.d_ = 1.0;
  b.d_ = 1.0;
  g.d_ = 1.0;
  fvar<double> dig_a = digamma(a);
  fvar<double> dig_b = digamma(b);
  fvar<double> dig_sum = digamma(a+b);
  fvar<double> beta_ab = exp(lbeta(a,b));
  fvar<double> g_a;
  fvar<double> g_b;

  stan::math::grad_reg_inc_beta(g_a,g_b,a, b, g,dig_a,dig_b,dig_sum,beta_ab);
  EXPECT_FLOAT_EQ(-0.36651629442883944183907601651838247842001142107486495485,
                  g_a.val_);
  EXPECT_NEAR(0.306495375042422864944011633197968575202046200428315551199,
              g_b.val_,1e-6);
}
TEST(ProbInternalMath, grad_reg_inc_beta_ffd) {
  using stan::math::fvar;
  using stan::math::digamma;
  using stan::math::exp;
  using stan::math::lbeta;

  fvar<fvar<double> > a = 1.0;
  fvar<fvar<double> > b = 1.0;
  fvar<fvar<double> > g = 0.4;
  a.d_ = 1.0;
  b.d_ = 1.0;
  g.d_ = 1.0;
  fvar<fvar<double> > dig_a = digamma(a);
  fvar<fvar<double> > dig_b = digamma(b);
  fvar<fvar<double> > dig_sum = digamma(a+b);
  fvar<fvar<double> > beta_ab = exp(lbeta(a,b));
  fvar<fvar<double> > g_a;
  fvar<fvar<double> > g_b;

  stan::math::grad_reg_inc_beta(g_a,g_b,a, b, g,dig_a,dig_b,dig_sum,beta_ab);
  EXPECT_FLOAT_EQ(-0.36651629442883944183907601651838247842001142107486495485,
                  g_a.val_.val_);
  EXPECT_NEAR(0.306495375042422864944011633197968575202046200428315551199,
              g_b.val_.val_,1e-6);
}
TEST(ProbInternalMath, grad_reg_inc_beta_fv) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::digamma;
  using stan::math::exp;
  using stan::math::lbeta;

  fvar<var> a = 1.0;
  fvar<var> b = 1.0;
  fvar<var> g = 0.4;
  a.d_ = 1.0;
  b.d_ = 1.0;
  g.d_ = 1.0;
  fvar<var> dig_a = digamma(a);
  fvar<var> dig_b = digamma(b);
  fvar<var> dig_sum = digamma(a+b);
  fvar<var> beta_ab = exp(lbeta(a,b));
  fvar<var> g_a;
  fvar<var> g_b;

  stan::math::grad_reg_inc_beta(g_a,g_b,a, b, g,dig_a,dig_b,dig_sum,beta_ab);
  EXPECT_FLOAT_EQ(-0.36651629442883944183907601651838247842001142107486495485,
                  g_a.val_.val());
  EXPECT_NEAR(0.306495375042422864944011633197968575202046200428315551199,
              g_b.val_.val(),1e-6);
}
TEST(ProbInternalMath, grad_reg_inc_beta_fv_1stDeriv1) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::digamma;
  using stan::math::exp;
  using stan::math::lbeta;

  fvar<var> a = 1.0;
  fvar<var> b = 1.0;
  fvar<var> g = 0.4;
  a.d_ = 1.0;
  b.d_ = 0.0;
  g.d_ = 0.0;
  fvar<var> dig_a = digamma(a);
  fvar<var> dig_b = digamma(b);
  fvar<var> dig_sum = digamma(a+b);
  fvar<var> beta_ab = exp(lbeta(a,b));
  fvar<var> g_a;
  fvar<var> g_b;

  stan::math::grad_reg_inc_beta(g_a,g_b,a, b, g,dig_a,dig_b,dig_sum,beta_ab);

  AVEC y1 = createAVEC(a.val_);
  VEC grad1;
  g_a.val_.grad(y1,grad1);
  EXPECT_FLOAT_EQ(0.33583548212738989400284958327902414335945097838423129,
                  grad1[0]);
}
TEST(ProbInternalMath, grad_reg_inc_beta_fv_1stDeriv2) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::digamma;
  using stan::math::exp;
  using stan::math::lbeta;

  fvar<var> a = 1.0;
  fvar<var> b = 1.0;
  fvar<var> g = 0.4;
  a.d_ = 0.0;
  b.d_ = 1.0;
  g.d_ = 0.0;
  fvar<var> dig_a = digamma(a);
  fvar<var> dig_b = digamma(b);
  fvar<var> dig_sum = digamma(a+b);
  fvar<var> beta_ab = exp(lbeta(a,b));
  fvar<var> g_a;
  fvar<var> g_b;

  stan::math::grad_reg_inc_beta(g_a,g_b,a, b, g,dig_a,dig_b,dig_sum,beta_ab);

  AVEC y1 = createAVEC(b.val_);
  VEC grad1;
  g_b.val_.grad(y1,grad1);
  EXPECT_NEAR(-0.156565690737548079304827886, grad1[0],1e-6);
}
TEST(ProbInternalMath, grad_reg_inc_beta_fv_2ndDeriv1) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::digamma;
  using stan::math::exp;
  using stan::math::lbeta;

  fvar<var> a = 1.0;
  fvar<var> b = 1.0;
  fvar<var> g = 0.4;
  a.d_ = 1.0;
  b.d_ = 0.0;
  g.d_ = 0.0;
  fvar<var> dig_a = digamma(a);
  fvar<var> dig_b = digamma(b);
  fvar<var> dig_sum = digamma(a+b);
  fvar<var> beta_ab = exp(lbeta(a,b));
  fvar<var> g_a;
  fvar<var> g_b;

  stan::math::grad_reg_inc_beta(g_a,g_b,a, b, g,dig_a,dig_b,dig_sum,beta_ab);

  AVEC y1 = createAVEC(a.val_);
  VEC grad1;
  g_a.d_.grad(y1,grad1);
  EXPECT_FLOAT_EQ(-0.30772293970781581317390510390046098438962772318921,
                  grad1[0]);
}
TEST(ProbInternalMath, grad_reg_inc_beta_fv_2ndDeriv2) {
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::digamma;
  using stan::math::exp;
  using stan::math::lbeta;

  fvar<var> a = 1.0;
  fvar<var> b = 1.0;
  fvar<var> g = 0.4;
  a.d_ = 0.0;
  b.d_ = 1.0;
  g.d_ = 0.0;
  fvar<var> dig_a = digamma(a);
  fvar<var> dig_b = digamma(b);
  fvar<var> dig_sum = digamma(a+b);
  fvar<var> beta_ab = exp(lbeta(a,b));
  fvar<var> g_a;
  fvar<var> g_b;

  stan::math::grad_reg_inc_beta(g_a,g_b,a, b, g,dig_a,dig_b,dig_sum,beta_ab);

  AVEC y1 = createAVEC(b.val_);
  VEC grad1;
  g_b.d_.grad(y1,grad1);
  EXPECT_NEAR(0.079977766631361187517939795, grad1[0],1e-4);
}

