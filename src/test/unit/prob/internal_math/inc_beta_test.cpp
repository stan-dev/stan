#include <gtest/gtest.h>
#include <stan/prob/internal_math/fwd/inc_beta.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <test/unit/agrad/util.hpp>

TEST(ProbInternalMath, inc_beta_fd) {
  using stan::agrad::fvar;
  fvar<double> a = 1.0;
  fvar<double> b = 1.0;
  fvar<double> g = 0.4;
  a.d_ = 1.0;
  b.d_ = 1.0;
  g.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(0.4, stan::agrad::inc_beta(a, b, g).val_);
  EXPECT_NEAR(0.1399790720133741432386770490663565874456006538094571344439, 
              stan::agrad::inc_beta(a, b, g).d_,1e-6);
}
TEST(ProbInternalMath, inc_beta_fv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> a = 1.0;
  fvar<var> b = 1.0;
  fvar<var> g = 0.4;
  a.d_ = 1.0;
  b.d_ = 1.0;
  g.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(0.4, stan::agrad::inc_beta(a, b, g).val_.val());
  EXPECT_NEAR(0.1399790720133741432386770490663565874456006538094571344439, 
              stan::agrad::inc_beta(a, b, g).d_.val(),1e-6);
}
TEST(ProbInternalMath, inc_beta_fv_2ndderiv1) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> a_fv = 1.0;
  fvar<var> b_fv = 1.0;
  fvar<var> g_fv = 0.4;
  a_fv.d_ = 0.0;
  b_fv.d_ = 0.0;
  g_fv.d_ = 1.0;
  
  fvar<var> z1 = stan::agrad::inc_beta(a_fv,b_fv,g_fv);

  AVEC y1 = createAVEC(g_fv.val_);
  VEC grad1;
  z1.d_.grad(y1,grad1);
  EXPECT_FLOAT_EQ(0, grad1[0]);
}
TEST(ProbInternalMath, inc_beta_fv_2ndderiv2) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> a_fv = 1.0;
  fvar<var> b_fv = 1.0;
  fvar<var> g_fv = 0.4;
  a_fv.d_ = 1.0;
  b_fv.d_ = 0.0;
  g_fv.d_ = 0.0;
  
  fvar<var> z1 = stan::agrad::inc_beta(a_fv,b_fv,g_fv);

  AVEC y1 = createAVEC(a_fv.val_);
  VEC grad1;
  z1.d_.grad(y1,grad1);
  EXPECT_FLOAT_EQ(1.8688680676267141432939094389587075968001762828102110959314, grad1[0]);
}
TEST(ProbInternalMath, inc_beta_fv_2ndderiv3) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> a_fv = 1.0;
  fvar<var> b_fv = 1.0;
  fvar<var> g_fv = 0.4;
  a_fv.d_ = 0.0;
  b_fv.d_ = 1.0;
  g_fv.d_ = 0.0;
  
  fvar<var> z1 = stan::agrad::inc_beta(a_fv,b_fv,g_fv);

  AVEC y1 = createAVEC(b_fv.val_);
  VEC grad1;
  z1.d_.grad(y1,grad1);
  EXPECT_NEAR(0.030443560743263100886529946935799949538080753702741024433088, grad1[0],1e-6);
}
  
TEST(ProbInternalMath, inc_beta_ffd) {
  using stan::agrad::fvar;
  fvar<fvar<double> > a = 1.0;
  fvar<fvar<double> > b = 1.0;
  fvar<fvar<double> > g = 0.4;
  a.d_ = 1.0;
  b.d_ = 1.0;
  g.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(0.4, stan::agrad::inc_beta(a, b, g).val_.val_);
  EXPECT_NEAR(0.1399790720133741432386770490663565874456006538094571344439, 
              stan::agrad::inc_beta(a, b, g).d_.val_,1e-6);
}

TEST(ProbInternalMath, inc_beta_fvv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > a = 1.0;
  fvar<fvar<var> > b = 1.0;
  fvar<fvar<var> > g = 0.4;
  a.d_ = 1.0;
  b.d_ = 1.0;
  g.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(0.4, stan::agrad::inc_beta(a, b, g).val_.val_.val());
  EXPECT_NEAR(0.1399790720133741432386770490663565874456006538094571344439, 
              stan::agrad::inc_beta(a, b, g).d_.val_.val(),1e-6);
}

TEST(ProbInternalMath, inc_beta_ffv_2ndderiv1) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > a_ffv = 1.0;
  fvar<fvar<var> > b_ffv = 1.0;
  fvar<fvar<var> > g_ffv = 0.4;
  a_ffv.d_ = 0.0;
  b_ffv.d_ = 0.0;
  g_ffv.d_ = 1.0;
  
  fvar<fvar<var> > z1 = stan::agrad::inc_beta(a_ffv,b_ffv,g_ffv);

  AVEC y1 = createAVEC(g_ffv.val_.val_);
  VEC grad1;
  z1.d_.val_.grad(y1,grad1);
  EXPECT_FLOAT_EQ(0, grad1[0]);
}
TEST(ProbInternalMath, inc_beta_ffv_2ndderiv2) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > a_ffv = 1.0;
  fvar<fvar<var> > b_ffv = 1.0;
  fvar<fvar<var> > g_ffv = 0.4;
  a_ffv.d_ = 1.0;
  b_ffv.d_ = 0.0;
  g_ffv.d_ = 0.0;
  
  fvar<fvar<var> > z1 = stan::agrad::inc_beta(a_ffv,b_ffv,g_ffv);

  AVEC y1 = createAVEC(a_ffv.val_.val_);
  VEC grad1;
  z1.d_.val_.grad(y1,grad1);
  EXPECT_FLOAT_EQ(1.8688680676267141432939094389587075968001762828102110959314,
                  grad1[0]);
}
TEST(ProbInternalMath, inc_beta_ffv_2ndderiv3) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > a_ffv = 1.0;
  fvar<fvar<var> > b_ffv = 1.0;
  fvar<fvar<var> > g_ffv = 0.4;
  a_ffv.d_ = 0.0;
  b_ffv.d_ = 1.0;
  g_ffv.d_ = 0.0;
  
  fvar<fvar<var> > z1 = stan::agrad::inc_beta(a_ffv,b_ffv,g_ffv);

  AVEC y1 = createAVEC(b_ffv.val_.val_);
  VEC grad1;
  z1.d_.val_.grad(y1,grad1);
  EXPECT_NEAR(0.030443560743263100886529946935799949538080753702741024433088,
              grad1[0],1e-6);
}
TEST(ProbInternalMath, inc_beta_ffv_3rddderiv1) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > a_ffv = 1.0;
  fvar<fvar<var> > b_ffv = 1.0;
  fvar<fvar<var> > g_ffv = 0.4;
  a_ffv.d_ = 0.0;
  b_ffv.d_ = 1.0;
  b_ffv.val_.d_ = 1.0;
  g_ffv.d_ = 0.0;
  
  fvar<fvar<var> > z1 = stan::agrad::inc_beta(a_ffv,b_ffv,g_ffv);

  AVEC y1 = createAVEC(b_ffv.val_.val_);
  VEC grad1;
  z1.d_.d_.grad(y1,grad1);
  EXPECT_NEAR(-0.01135291559842811507560643721669539256593397382680184269398, 
              grad1[0],1e-6);
}
TEST(ProbInternalMath, inc_beta_ffv_3rddderiv2) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > a_ffv = 1.0;
  fvar<fvar<var> > b_ffv = 1.0;
  fvar<fvar<var> > g_ffv = 0.4;
  a_ffv.d_ = 1.0;
  a_ffv.val_.d_ = 1.0;
  b_ffv.d_ = 0.0;
  g_ffv.d_ = 0.0;
  
  fvar<fvar<var> > z1 = stan::agrad::inc_beta(a_ffv,b_ffv,g_ffv);

  AVEC y1 = createAVEC(a_ffv.val_.val_);
  VEC grad1;
  z1.d_.d_.grad(y1,grad1);
  EXPECT_NEAR(-5.91432714258796063698285821989494279917844039011638473858017, 
              grad1[0],1e-6);
}
TEST(ProbInternalMath, inc_beta_ffv_3rddderiv3) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > a_ffv = 1.0;
  fvar<fvar<var> > b_ffv = 1.0;
  fvar<fvar<var> > g_ffv = 0.4;
  a_ffv.d_ = 0.0;
  b_ffv.d_ = 0.0;
  g_ffv.d_ = 1.0;
  g_ffv.val_.d_ = 1.0;
  
  fvar<fvar<var> > z1 = stan::agrad::inc_beta(a_ffv,b_ffv,g_ffv);

  AVEC y1 = createAVEC(g_ffv.val_.val_);
  VEC grad1;
  z1.d_.d_.grad(y1,grad1);
  EXPECT_FLOAT_EQ(0, grad1[0]);
}
  
