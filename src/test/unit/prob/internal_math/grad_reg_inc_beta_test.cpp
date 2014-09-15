#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/prob/internal_math/fwd/inc_beta.hpp>
#include <stan/prob/internal_math/rev/inc_beta.hpp>
#include <stan/prob/internal_math/math/grad_reg_inc_beta.hpp>
#include <test/unit/agrad/util.hpp>

TEST(ProbInternalMath, grad_reg_inc_beta_fd) {
  using stan::agrad::fvar;
  using stan::agrad::digamma;
  using stan::agrad::exp;
  using stan::agrad::lbeta;

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
  using stan::agrad::fvar;
  using stan::agrad::digamma;
  using stan::agrad::exp;
  using stan::agrad::lbeta;

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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::digamma;
  using stan::agrad::exp;
  using stan::agrad::lbeta;

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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::digamma;
  using stan::agrad::exp;
  using stan::agrad::lbeta;

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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::digamma;
  using stan::agrad::exp;
  using stan::agrad::lbeta;

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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::digamma;
  using stan::agrad::exp;
  using stan::agrad::lbeta;

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
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::agrad::digamma;
  using stan::agrad::exp;
  using stan::agrad::lbeta;

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

