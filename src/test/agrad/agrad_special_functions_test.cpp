#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <boost/math/special_functions.hpp>
#include <gtest/gtest.h>
#include "stan/agrad/agrad.hpp"
#include "stan/agrad/special_functions.hpp"


// cut and paste helpers and typedefs from agrad_test.cpp
typedef stan::agrad::var AVAR;
typedef std::vector<AVAR> AVEC;
typedef std::vector<double> VEC;

AVEC createAVEC(AVAR x) {
  AVEC v;
  v.push_back(x);
  return v;
}
AVEC createAVEC(AVAR x1, AVAR x2) {
  AVEC v;
  v.push_back(x1);
  v.push_back(x2);
  return v;
}
AVEC createAVEC(AVAR x1, AVAR x2, AVAR x3) {
  AVEC v;
  v.push_back(x1);
  v.push_back(x2);
  v.push_back(x3);
  return v;
}
// end cut-and-paste

TEST(agrad_agrad_special_functions,lgamma) {
  AVAR a = 3.0;
  AVAR f = lgamma(a);
  EXPECT_FLOAT_EQ(lgamma(3.0),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(boost::math::digamma(3.0),grad_f[0]);
}

TEST(agrad_agrad_special_functions,fma_vvv) {
  AVAR a = 3.0;
  AVAR b = 5.0;
  AVAR c = 7.0;
  AVAR f = fma(a,b,c);
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, f.val());
  
  AVEC x = createAVEC(a,b,c);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(5.0,grad_f[0]);
  EXPECT_FLOAT_EQ(3.0,grad_f[1]);
  EXPECT_FLOAT_EQ(1.0,grad_f[2]);
}  

TEST(agrad_agrad_special_functions,fma_vvd) {
  AVAR a = 3.0;
  AVAR b = 5.0;
  double c = 7.0;
  AVAR f = fma(a,b,c);
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, f.val());
  
  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(5.0,grad_f[0]);
  EXPECT_FLOAT_EQ(3.0,grad_f[1]);
}  

TEST(agrad_agrad_special_functions,fma_vdv) {
  AVAR a = 3.0;
  double b = 5.0;
  AVAR c = 7.0;
  AVAR f = fma(a,b,c);
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, f.val());
  
  AVEC x = createAVEC(a,c);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(5.0,grad_f[0]);
  EXPECT_FLOAT_EQ(1.0,grad_f[1]);
}  

TEST(agrad_agrad_special_functions,fma_vdd) {
  AVAR a = 3.0;
  double b = 5.0;
  double c = 7.0;
  AVAR f = fma(a,b,c);
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(5.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fma_dvv) {
  double a = 3.0;
  AVAR b = 5.0;
  AVAR c = 7.0;
  AVAR f = fma(a,b,c);
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, f.val());
  
  AVEC x = createAVEC(b,c);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(3.0,grad_f[0]);
  EXPECT_FLOAT_EQ(1.0,grad_f[1]);
}  

TEST(agrad_agrad_special_functions,fma_dvd) {
  double a = 3.0;
  AVAR b = 5.0;
  double c = 7.0;
  AVAR f = fma(a,b,c);
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, f.val());
  
  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(3.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fma_ddv) {
  double a = 3.0;
  double b = 5.0;
  AVAR c = 7.0;
  AVAR f = fma(a,b,c);
  EXPECT_FLOAT_EQ(3.0 * 5.0 + 7.0, f.val());
  
  AVEC x = createAVEC(c);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  


TEST(agrad_agrad_special_functions,inv_logit) {
  AVAR a = 2.0;
  AVAR f = inv_logit(a);
  EXPECT_FLOAT_EQ(1.0 / (1.0 + exp(-2.0)),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(exp(-2.0)/pow(1 + exp(-2.0),2.0),
		  grad_f[0]);
}

TEST(agrad_agrad_special_functions,log1p) {
  AVAR a = 0.1;
  AVAR f = log1p(a);
  EXPECT_FLOAT_EQ(log(1 + 0.1), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0 / (1.0 + 0.1), grad_f[0]);
}

TEST(agrad_agrad_special_functions,log1m) {
  AVAR a = 0.1;
  AVAR f = log1m(a);
  EXPECT_FLOAT_EQ(log(1 - 0.1), f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-1.0/(1.0 - 0.1), grad_f[0]);
}

TEST(agrad_agrad_special_functions,log_loss_zero) {
  AVAR y_hat = 0.2;
  int y = 0;
  AVAR f = log_loss(y,y_hat);
  EXPECT_FLOAT_EQ(-log(1.0 - 0.2), f.val());

  AVEC x = createAVEC(y_hat);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ((1.0 / (1.0 - 0.2)), grad_f[0]);
}

TEST(agrad_agrad_special_functions,log_loss_one) {
  AVAR y_hat = 0.2;
  int y = 1;
  AVAR f = log_loss(y,y_hat);
  EXPECT_FLOAT_EQ(-log(0.2), f.val());

  AVEC x = createAVEC(y_hat);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-1.0 / 0.2, grad_f[0]);
}

TEST(agrad_agrad_special_functions,acosh) {
  AVAR a = 1.3;
  AVAR f = acosh(a);
  EXPECT_FLOAT_EQ(acosh(1.3), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0/sqrt(1.3 * 1.3  - 1.0), grad_f[0]);
}

TEST(agrad_agrad_special_functions,asinh) {
  AVAR a = 0.2;
  AVAR f = asinh(a);
  EXPECT_FLOAT_EQ(asinh(0.2), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0/sqrt(0.2 * 0.2  + 1.0), grad_f[0]);
}

TEST(agrad_agrad_special_functions,atanh) {
  AVAR a = 0.3;
  AVAR f = atanh(a);
  EXPECT_FLOAT_EQ(atanh(0.3), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0/(1.0 - 0.3 * 0.3), grad_f[0]);
}

TEST(agrad_agrad_special_functions,erf) {
  AVAR a = 1.3;
  AVAR f = erf(a);
  EXPECT_FLOAT_EQ(boost::math::erf(1.3), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(2.0 / std::sqrt(boost::math::constants::pi<double>()) * std::exp(- 1.3 * 1.3), grad_f[0]);
}

TEST(agrad_agrad_special_functions,erfc) {
  AVAR a = 1.3;
  AVAR f = erfc(a);
  EXPECT_FLOAT_EQ(boost::math::erfc(1.3), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-2.0 / std::sqrt(boost::math::constants::pi<double>()) * std::exp(- 1.3 * 1.3), grad_f[0]);
}
  

TEST(agrad_agrad_special_functions,exp2) {
  AVAR a = 1.3;
  AVAR f = exp2(a);
  EXPECT_FLOAT_EQ(std::pow(2.0,1.3), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::pow(2.0,1.3) * std::log(2.0),grad_f[0]);
}

TEST(agrad_agrad_special_functions,expm1) {
  AVAR a = 1.3;
  AVAR f = expm1(a);
  EXPECT_FLOAT_EQ(boost::math::expm1(1.3), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::exp(1.3) - 1.0, grad_f[0]);
}  


TEST(agrad_agrad_special_functions,fmax_vv) {
  AVAR a = 1.3;
  AVAR b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
  EXPECT_FLOAT_EQ(1.0,grad_f[1]);
}  

TEST(agrad_agrad_special_functions,fmax_vv_2) {
  AVAR a = 2.3;
  AVAR b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.3,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
  EXPECT_FLOAT_EQ(0.0,grad_f[1]);
}  

TEST(agrad_agrad_special_functions,fmax_vv_3) {
  AVAR a = 2.0;
  AVAR b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but documented this way
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
  EXPECT_FLOAT_EQ(1.0,grad_f[1]);
}  

TEST(agrad_agrad_special_functions,fmax_vd) {
  AVAR a = 1.3;
  double b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fmax_vd_2) {
  AVAR a = 2.3;
  double b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.3,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fmax_vd_3) {
  AVAR a = 2.0;
  double b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fmax_dv) {
  double a = 1.3;
  AVAR b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fmax_dv_2) {
  double a = 2.3;
  AVAR b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.3,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fmax_dv_3) {
  double a = 2.0;
  AVAR b = 2.0;
  AVAR f = fmax(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but doc this way
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fmin_vv) {
  AVAR a = 1.3;
  AVAR b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(1.3,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
  EXPECT_FLOAT_EQ(0.0,grad_f[1]);
}  

TEST(agrad_agrad_special_functions,fmin_vv_2) {
  AVAR a = 2.3;
  AVAR b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
  EXPECT_FLOAT_EQ(1.0,grad_f[1]);
}  

TEST(agrad_agrad_special_functions,fmin_vv_3) {
  AVAR a = 2.0;
  AVAR b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but documented this way
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
  EXPECT_FLOAT_EQ(1.0,grad_f[1]);
}  

TEST(agrad_agrad_special_functions,fmin_vd) {
  AVAR a = 1.3;
  double b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(1.3,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fmin_vd_2) {
  AVAR a = 2.3;
  double b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fmin_vd_3) {
  AVAR a = 2.0;
  double b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fmin_dv) {
  double a = 1.3;
  AVAR b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(1.3,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fmin_dv_2) {
  double a = 2.3;
  AVAR b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fmin_dv_3) {
  double a = 2.0;
  AVAR b = 2.0;
  AVAR f = fmin(a,b);
  EXPECT_FLOAT_EQ(2.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but doc this way
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,hypot_vv) {
  AVAR a = 3.0;
  AVAR b = 4.0;
  AVAR f = hypot(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but doc this way
  EXPECT_FLOAT_EQ(3.0/5.0,grad_f[0]);
  EXPECT_FLOAT_EQ(4.0/5.0,grad_f[1]);
}  

TEST(agrad_agrad_special_functions,hypot_vd) {
  AVAR a = 3.0;
  double b = 4.0;
  AVAR f = hypot(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but doc this way
  EXPECT_FLOAT_EQ(3.0/5.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,hypot_dv) {
  double a = 3.0;
  AVAR b = 4.0;
  AVAR f = hypot(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  // arbitrary, but doc this way
  EXPECT_FLOAT_EQ(4.0/5.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,log2) {
  AVAR a = 3.0;
  AVAR f = log2(a);
  EXPECT_FLOAT_EQ(std::log(3.0)/std::log(2.0), f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0 / 3.0 / std::log(2.0), grad_f[0]);
}

TEST(agrad_agrad_special_functions,cbrt) {
  AVAR a = 27.0;
  AVAR f = cbrt(a);
  EXPECT_FLOAT_EQ(3.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0 / 3.0 / std::pow(27.0,2.0/3.0), grad_f[0]);
}

TEST(agrad_agrad_special_functions,trunc) {
  AVAR a = 1.2;
  AVAR f = trunc(a);
  EXPECT_FLOAT_EQ(1.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0, grad_f[0]);
}

TEST(agrad_agrad_special_functions,trunc_2) {
  AVAR a = -1.2;
  AVAR f = trunc(a);
  EXPECT_FLOAT_EQ(-1.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0, grad_f[0]);
}

TEST(agrad_agrad_special_functions,round) {
  AVAR a = 1.2;
  AVAR f = round(a);
  EXPECT_FLOAT_EQ(1.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0, grad_f[0]);
}

TEST(agrad_agrad_special_functions,round_2) {
  AVAR a = -1.2;
  AVAR f = round(a);
  EXPECT_FLOAT_EQ(-1.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0, grad_f[0]);
}

TEST(agrad_agrad_special_functions,round_3) {
  AVAR a = 1.7;
  AVAR f = round(a);
  EXPECT_FLOAT_EQ(2.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0, grad_f[0]);
}


TEST(agrad_agrad_special_functions,round_4) {
  AVAR a = -1.7;
  AVAR f = round(a);
  EXPECT_FLOAT_EQ(-2.0, f.val());
  
  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0, grad_f[0]);
}

TEST(agrad_agrad_special_functions,fdim_vv) {
  AVAR a = 3.0;
  AVAR b = 4.0;
  AVAR f = fdim(a,b);
  EXPECT_FLOAT_EQ(0.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
  EXPECT_FLOAT_EQ(0.0,grad_f[1]);
}  

TEST(agrad_agrad_special_functions,fdim_vv_2) {
  AVAR a = 7.0;
  AVAR b = 2.0;
  AVAR f = fdim(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(a,b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
  EXPECT_FLOAT_EQ(-1.0,grad_f[1]);
}  

TEST(agrad_agrad_special_functions,fdim_vd) {
  AVAR a = 3.0;
  double b = 4.0;
  AVAR f = fdim(a,b);
  EXPECT_FLOAT_EQ(0.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fdim_vd_2) {
  AVAR a = 7.0;
  double b = 2.0;
  AVAR f = fdim(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(1.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fdim_dv) {
  double a = 3.0;
  AVAR b = 4.0;
  AVAR f = fdim(a,b);
  EXPECT_FLOAT_EQ(0.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,fdim_dv_2) {
  double a = 7.0;
  AVAR b = 2.0;
  AVAR f = fdim(a,b);
  EXPECT_FLOAT_EQ(5.0,f.val());

  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-1.0,grad_f[0]);
}  
 

TEST(agrad_agrad_special_functions,tgamma) {
  AVAR a = 3.5;
  AVAR f = tgamma(a);
  EXPECT_FLOAT_EQ(boost::math::tgamma(3.5),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(boost::math::digamma(3.5) * boost::math::tgamma(3.5),grad_f[0]);
}  

TEST(agrad_agrad_special_functions,step) {
  AVAR a = 3.5;
  AVAR f = step(a);
  EXPECT_FLOAT_EQ(1.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,step_2) {
  AVAR a = 0.0;
  AVAR f = step(a);
  EXPECT_FLOAT_EQ(1.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  

TEST(agrad_agrad_special_functions,step_3) {
  AVAR a = -18765.3;
  AVAR f = step(a);
  EXPECT_FLOAT_EQ(0.0,f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(0.0,grad_f[0]);
}  
 
TEST(agrad_agrad_special_functions,inv_cloglog) {
  AVAR a = 2.7;
  AVAR f = inv_cloglog(a);
  EXPECT_FLOAT_EQ(std::exp(-std::exp(2.7)),f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(-std::exp(2.7 - std::exp(2.7)),grad_f[0]);
}












