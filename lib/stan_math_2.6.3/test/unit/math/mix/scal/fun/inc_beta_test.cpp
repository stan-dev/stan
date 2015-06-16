#include <gtest/gtest.h>
#include <stan/math/fwd/scal/fun/inc_beta.hpp>
#include <stan/math/rev/scal/fun/inc_beta.hpp>
#include <stan/math/rev/scal/fun/grad_inc_beta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/digamma.hpp>
#include <stan/math/rev/scal/fun/digamma.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/rev/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/ceil.hpp>
#include <stan/math/rev/scal/fun/ceil.hpp>

TEST(ProbInternalMath, inc_beta_fd) {
  using stan::math::fvar;
  fvar<double> a = 1.0;
  fvar<double> b = 1.0;
  fvar<double> g = 0.4;
  a.d_ = 1.0;
  b.d_ = 1.0;
  g.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(0.4, stan::math::inc_beta(a, b, g).val_);
  EXPECT_NEAR(-0.36651629442883944183907601651838247842001142107486495485 
              + 0.306495375042422864944011633197968575202046200428315551199
              + std::pow(1-0.4,1.0-1.0)*std::pow(0.4,1.0-1.0) 
                / std::exp(stan::math::lbeta(1.0,1.0)),
              stan::math::inc_beta(a, b, g).d_,1e-6);
}
TEST(ProbInternalMath, inc_beta_fv) {
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> a = 1.0;
  fvar<var> b = 1.0;
  fvar<var> g = 0.4;
  a.d_ = 1.0;
  b.d_ = 1.0;
  g.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(0.4, stan::math::inc_beta(a, b, g).val_.val());
  EXPECT_NEAR(-0.36651629442883944183907601651838247842001142107486495485 
              + 0.306495375042422864944011633197968575202046200428315551199
              + std::pow(1-0.4,1.0-1.0)*std::pow(0.4,1.0-1.0) 
                / std::exp(stan::math::lbeta(1.0,1.0)), 
              stan::math::inc_beta(a, b, g).d_.val(),1e-6);
}
TEST(ProbInternalMath, inc_beta_fv_2ndderiv1) {
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> a_fv = 1.0;
  fvar<var> b_fv = 1.0;
  fvar<var> g_fv = 0.4;
  a_fv.d_ = 0.0;
  b_fv.d_ = 0.0;
  g_fv.d_ = 1.0;
  
  fvar<var> z1 = stan::math::inc_beta(a_fv,b_fv,g_fv);

  AVEC y1 = createAVEC(g_fv.val_);
  VEC grad1;
  z1.d_.grad(y1,grad1);
  EXPECT_FLOAT_EQ(0, grad1[0]);
}
TEST(ProbInternalMath, inc_beta_fv_2ndderiv2) {
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> a_fv = 1.0;
  fvar<var> b_fv = 1.0;
  fvar<var> g_fv = 0.4;
  a_fv.d_ = 1.0;
  b_fv.d_ = 0.0;
  g_fv.d_ = 0.0;
  
  fvar<var> z1 = stan::math::inc_beta(a_fv,b_fv,g_fv);

  AVEC y1 = createAVEC(a_fv.val_);
  VEC grad1;
  z1.d_.grad(y1,grad1);
  EXPECT_FLOAT_EQ(0.335835482127389894002849583279024143359450978384231290056028, grad1[0]);
}
TEST(ProbInternalMath, inc_beta_fv_2ndderiv3) {
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> a_fv = 1.0;
  fvar<var> b_fv = 1.0;
  fvar<var> g_fv = 0.4;
  a_fv.d_ = 0.0;
  b_fv.d_ = 1.0;
  g_fv.d_ = 0.0;
  
  fvar<var> z1 = stan::math::inc_beta(a_fv,b_fv,g_fv);

  AVEC y1 = createAVEC(b_fv.val_);
  VEC grad1;
  z1.d_.grad(y1,grad1);
  EXPECT_NEAR(-0.15656569077159365641351913104510130858703903615614356859968,
              grad1[0],1e-6);
}
  
TEST(ProbInternalMath, inc_beta_ffd) {
  using stan::math::fvar;
  fvar<fvar<double> > a = 1.0;
  fvar<fvar<double> > b = 1.0;
  fvar<fvar<double> > g = 0.4;
  a.d_ = 1.0;
  b.d_ = 1.0;
  g.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(0.4, stan::math::inc_beta(a, b, g).val_.val_);
  EXPECT_NEAR(-0.36651629442883944183907601651838247842001142107486495485 
              + 0.306495375042422864944011633197968575202046200428315551199
              + std::pow(1-0.4,1.0-1.0)*std::pow(0.4,1.0-1.0) 
                / std::exp(stan::math::lbeta(1.0,1.0)), 
              stan::math::inc_beta(a, b, g).d_.val_,1e-6);
}

TEST(ProbInternalMath, inc_beta_fvv) {
  using stan::math::fvar;
  using stan::math::var;
  fvar<fvar<var> > a = 1.0;
  fvar<fvar<var> > b = 1.0;
  fvar<fvar<var> > g = 0.4;
  a.d_ = 1.0;
  b.d_ = 1.0;
  g.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(0.4, stan::math::inc_beta(a, b, g).val_.val_.val());
  EXPECT_NEAR(-0.36651629442883944183907601651838247842001142107486495485 
              + 0.306495375042422864944011633197968575202046200428315551199
              + std::pow(1-0.4,1.0-1.0)*std::pow(0.4,1.0-1.0) 
                / std::exp(stan::math::lbeta(1.0,1.0)), 
              stan::math::inc_beta(a, b, g).d_.val_.val(),1e-6);
}

TEST(ProbInternalMath, inc_beta_ffv_2ndderiv1) {
  using stan::math::fvar;
  using stan::math::var;
  fvar<fvar<var> > a_ffv = 1.0;
  fvar<fvar<var> > b_ffv = 1.0;
  fvar<fvar<var> > g_ffv = 0.4;
  a_ffv.d_ = 0.0;
  b_ffv.d_ = 0.0;
  g_ffv.d_ = 1.0;
  
  fvar<fvar<var> > z1 = stan::math::inc_beta(a_ffv,b_ffv,g_ffv);

  AVEC y1 = createAVEC(g_ffv.val_.val_);
  VEC grad1;
  z1.d_.val_.grad(y1,grad1);
  EXPECT_FLOAT_EQ(0, grad1[0]);
}
TEST(ProbInternalMath, inc_beta_ffv_2ndderiv2) {
  using stan::math::fvar;
  using stan::math::var;
  fvar<fvar<var> > a_ffv = 1.0;
  fvar<fvar<var> > b_ffv = 1.0;
  fvar<fvar<var> > g_ffv = 0.4;
  a_ffv.d_ = 1.0;
  b_ffv.d_ = 0.0;
  g_ffv.d_ = 0.0;
  
  fvar<fvar<var> > z1 = stan::math::inc_beta(a_ffv,b_ffv,g_ffv);

  AVEC y1 = createAVEC(a_ffv.val_.val_);
  VEC grad1;
  z1.d_.val_.grad(y1,grad1);
  EXPECT_FLOAT_EQ(0.335835482127389894002849583279024143359450978384231290056028,
                  grad1[0]);
}
TEST(ProbInternalMath, inc_beta_ffv_2ndderiv3) {
  using stan::math::fvar;
  using stan::math::var;
  fvar<fvar<var> > a_ffv = 1.0;
  fvar<fvar<var> > b_ffv = 1.0;
  fvar<fvar<var> > g_ffv = 0.4;
  a_ffv.d_ = 0.0;
  b_ffv.d_ = 1.0;
  g_ffv.d_ = 0.0;
  
  fvar<fvar<var> > z1 = stan::math::inc_beta(a_ffv,b_ffv,g_ffv);

  AVEC y1 = createAVEC(b_ffv.val_.val_);
  VEC grad1;
  z1.d_.val_.grad(y1,grad1);
  EXPECT_NEAR(-0.15656569077159365641351913104510130858703903615614356859968,
              grad1[0],1e-6);
}
TEST(ProbInternalMath, inc_beta_ffv_3rddderiv1) {
  using stan::math::fvar;
  using stan::math::var;
  fvar<fvar<var> > a_ffv = 1.0;
  fvar<fvar<var> > b_ffv = 1.0;
  fvar<fvar<var> > g_ffv = 0.4;
  a_ffv.d_ = 0.0;
  b_ffv.d_ = 1.0;
  b_ffv.val_.d_ = 1.0;
  g_ffv.d_ = 0.0;
  
  fvar<fvar<var> > z1 = stan::math::inc_beta(a_ffv,b_ffv,g_ffv);

  AVEC y1 = createAVEC(b_ffv.val_.val_);
  VEC grad1;
  z1.d_.d_.grad(y1,grad1);
  EXPECT_NEAR(0.079976746033671442, 
              grad1[0],1e-6);
}
TEST(ProbInternalMath, inc_beta_ffv_3rddderiv2) {
  using stan::math::fvar;
  using stan::math::var;
  fvar<fvar<var> > a_ffv = 1.0;
  fvar<fvar<var> > b_ffv = 1.0;
  fvar<fvar<var> > g_ffv = 0.4;
  a_ffv.d_ = 1.0;
  a_ffv.val_.d_ = 1.0;
  b_ffv.d_ = 0.0;
  g_ffv.d_ = 0.0;
  
  fvar<fvar<var> > z1 = stan::math::inc_beta(a_ffv,b_ffv,g_ffv);

  AVEC y1 = createAVEC(a_ffv.val_.val_);
  VEC grad1;
  z1.d_.d_.grad(y1,grad1);
  EXPECT_NEAR(-0.30772293970781581317390510390046098438962772318921188609907, 
              grad1[0],1e-6);
}
TEST(ProbInternalMath, inc_beta_ffv_3rddderiv3) {
  using stan::math::fvar;
  using stan::math::var;
  fvar<fvar<var> > a_ffv = 1.0;
  fvar<fvar<var> > b_ffv = 1.0;
  fvar<fvar<var> > g_ffv = 0.4;
  a_ffv.d_ = 0.0;
  b_ffv.d_ = 0.0;
  g_ffv.d_ = 1.0;
  g_ffv.val_.d_ = 1.0;
  
  fvar<fvar<var> > z1 = stan::math::inc_beta(a_ffv,b_ffv,g_ffv);

  AVEC y1 = createAVEC(g_ffv.val_.val_);
  VEC grad1;
  z1.d_.d_.grad(y1,grad1);
  EXPECT_FLOAT_EQ(0, grad1[0]);
}
  
