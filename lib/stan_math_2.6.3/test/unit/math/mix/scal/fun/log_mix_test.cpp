#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/fun/log_mix.hpp>
#include <stan/math/prim/mat/err/constraint_tolerance.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <cmath>
#include <typeinfo>
#include <stan/math/fwd/scal/fun/log_mix.hpp>
#include <stan/math/rev/scal/fun/log_mix.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/rev/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>

void test_log_mix_3xfvar_var_D1(double theta,
    double lambda1, double lambda2, double theta_d, 
    double lambda1_d, double lambda2_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<var> theta_fv(theta,theta_d);
  fvar<var> lambda1_fv(lambda1,lambda1_d);
  fvar<var> lambda2_fv(lambda2,lambda2_d);

  var theta_v(theta);
  var lambda1_v(lambda1);
  var lambda2_v(lambda2);

  var theta_d_v(theta_d);
  var lambda1_d_v(lambda1_d);
  var lambda2_d_v(lambda2_d);

  fvar<var> res = log_mix(theta_fv, lambda1_fv, lambda2_fv);
  double result = log_mix(theta_fv.val_.val(), lambda1_fv.val_.val(),
                          lambda2_fv.val_.val());
  double deriv_denom = exp(lambda1_fv.val_.val()) * theta_fv.val_.val()
    + exp(lambda2_fv.val_.val()) * (1 - theta_fv.val_.val());
  double theta_deriv = 1 / deriv_denom * (exp(lambda1_fv.val_.val()) 
        - exp(lambda2_fv.val_.val()));
  double lambda1_deriv = 1 / deriv_denom * exp(lambda1_fv.val_.val())
          * theta_fv.val_.val();
  double lambda2_deriv = 1 / deriv_denom * exp(lambda2_fv.val_.val()) 
    * (1 - theta_fv.val_.val());
  double deriv = theta_deriv * theta_fv.d_.val()
             + lambda2_deriv * lambda2_fv.d_.val() 
             + lambda1_deriv * lambda1_fv.d_.val();

  VEC g = cgrad(res.val_,theta_fv.val_, lambda1_fv.val_,
     lambda2_fv.val_);

  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(theta_deriv,g[0]);
  EXPECT_FLOAT_EQ(lambda1_deriv,g[1]);
  EXPECT_FLOAT_EQ(lambda2_deriv,g[2]);

  fvar<var> theta_fv_invalid(-1.0,theta_d);
  EXPECT_THROW(log_mix(theta_fv_invalid,lambda1_fv,lambda2_fv),std::domain_error);
  stan::math::recover_memory();
}

VEC log_mix_D3(double theta, double lambda1, double lambda2,
               double theta_d, double lambda1_d, double lambda2_d,
               double theta_d2, double lambda1_d2, double lambda2_d2){
  using stan::math::log_mix;
  using std::exp;
  using std::pow;
  using stan::math::var;

  var theta_v(theta);
  var lambda1_v(lambda1);
  var lambda2_v(lambda2);
  double d_theta(0.0);
  double d_lambda1(0.0);
  double d_lambda2(0.0);
  double d2_theta(0.0);
  double d2_lambda1(0.0);
  double d2_lambda2(0.0);
  var d_theta_v;
  var d_lambda1_v;
  var d_lambda2_v;
  var d2_theta_v;
  var d2_lambda1_v;
  var d2_lambda2_v;
  if (lambda1 > lambda2) {
        double lam2_m_lam1 = lambda2 - lambda1;
        double exp_lam2_m_lam1 = exp(lam2_m_lam1);
        double one_m_exp_lam2_m_lam1 = 1 - exp_lam2_m_lam1;
        double one_m_t = 1 - theta;
        double one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        double t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta + one_m_t_prod_exp_lam2_m_lam1;
        var lam2_m_lam1_v = lambda2_v - lambda1_v;
        var exp_lam2_m_lam1_v = exp(lam2_m_lam1_v);
        var one_m_exp_lam2_m_lam1_v = 1 - exp_lam2_m_lam1_v;
        var one_m_t_v = 1 - theta_v;
        var one_m_t_prod_exp_lam2_m_lam1_v = one_m_t_v * exp_lam2_m_lam1_v;
        var t_plus_one_m_t_prod_exp_lam2_m_lam1_v 
          = theta_v + one_m_t_prod_exp_lam2_m_lam1_v;
        d_theta 
          = one_m_exp_lam2_m_lam1 
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
        d_lambda1
          = theta
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
        d_lambda2
          = one_m_t_prod_exp_lam2_m_lam1
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
        d_theta_v 
          = one_m_exp_lam2_m_lam1_v 
          / t_plus_one_m_t_prod_exp_lam2_m_lam1_v;
        d_lambda1_v
          = theta_v
          / t_plus_one_m_t_prod_exp_lam2_m_lam1_v;
        d_lambda2_v
          = one_m_t_prod_exp_lam2_m_lam1_v
          / t_plus_one_m_t_prod_exp_lam2_m_lam1_v;
        d2_theta
          = lambda1_d * (1.0 / t_plus_one_m_t_prod_exp_lam2_m_lam1 - d_lambda1 * d_theta)
          - lambda2_d * (exp_lam2_m_lam1 / t_plus_one_m_t_prod_exp_lam2_m_lam1 + d_lambda2 * d_theta)
          - theta_d * pow(d_theta,2.0);
        d2_lambda1 
          = lambda1_d * (d_lambda1 - pow(d_lambda1,2.0))
          - lambda2_d * d_lambda1 * d_lambda2
          + theta_d * (1.0 / t_plus_one_m_t_prod_exp_lam2_m_lam1 - d_lambda1 * d_theta);
        d2_lambda2 
          = lambda2_d * (d_lambda2 - pow(d_lambda2,2.0)) 
          - lambda1_d * d_lambda1 * d_lambda2
          - theta_d * (d_lambda2 * d_theta + exp_lam2_m_lam1 / t_plus_one_m_t_prod_exp_lam2_m_lam1);
        d2_theta_v
          = lambda1_d * (1.0 / t_plus_one_m_t_prod_exp_lam2_m_lam1_v - d_lambda1_v * d_theta_v)
          - lambda2_d * (exp_lam2_m_lam1_v / t_plus_one_m_t_prod_exp_lam2_m_lam1_v + d_lambda2_v * d_theta_v)
          - theta_d * pow(d_theta_v,2.0);
        d2_lambda1_v 
          = lambda1_d * (d_lambda1_v - pow(d_lambda1_v,2.0))
          - lambda2_d * d_lambda1_v * d_lambda2_v
          + theta_d * (1.0 / t_plus_one_m_t_prod_exp_lam2_m_lam1_v - d_lambda1_v * d_theta_v);
        d2_lambda2_v 
          = lambda2_d * (d_lambda2_v - pow(d_lambda2_v,2.0)) 
          - lambda1_d * d_lambda1_v * d_lambda2_v
          - theta_d * (d_lambda2_v * d_theta_v + exp_lam2_m_lam1_v / t_plus_one_m_t_prod_exp_lam2_m_lam1_v);
      } else {
        double lam1_m_lam2 = lambda1 - lambda2;
        double exp_lam1_m_lam2 = exp(lam1_m_lam2);
        double exp_lam1_m_lam2_m_1 = exp_lam1_m_lam2 - 1;
        double one_m_t = 1 - theta;
        double t_prod_exp_lam1_m_lam2 = theta * exp_lam1_m_lam2;
        double one_m_t_plus_t_prod_exp_lam1_m_lam2 
          = one_m_t + t_prod_exp_lam1_m_lam2;
        var lam1_m_lam2_v = lambda1_v - lambda2_v;
        var exp_lam1_m_lam2_v = exp(lam1_m_lam2_v);
        var exp_lam1_m_lam2_m_1_v = exp_lam1_m_lam2_v - 1;
        var one_m_t_v = 1 - theta_v;
        var t_prod_exp_lam1_m_lam2_v = theta_v * exp_lam1_m_lam2_v;
        var one_m_t_plus_t_prod_exp_lam1_m_lam2_v 
          = one_m_t_v + t_prod_exp_lam1_m_lam2_v;
        d_theta 
          = exp_lam1_m_lam2_m_1
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
        d_lambda1
          = t_prod_exp_lam1_m_lam2
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
        d_lambda2
          = one_m_t
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
        d_theta_v 
          = exp_lam1_m_lam2_m_1_v
          / one_m_t_plus_t_prod_exp_lam1_m_lam2_v;
        d_lambda1_v
          = t_prod_exp_lam1_m_lam2_v
          / one_m_t_plus_t_prod_exp_lam1_m_lam2_v;
        d_lambda2_v
          = one_m_t_v
          / one_m_t_plus_t_prod_exp_lam1_m_lam2_v;
        d2_theta 
          = lambda1_d * (exp_lam1_m_lam2 / one_m_t_plus_t_prod_exp_lam1_m_lam2 - d_lambda1 * d_theta)
          - lambda2_d * (1.0 / one_m_t_plus_t_prod_exp_lam1_m_lam2 + d_lambda2 * d_theta)
          - theta_d * pow(d_theta,2.0);
        d2_lambda1 
          = lambda1_d * (d_lambda1 - pow(d_lambda1,2.0))
          - lambda2_d * d_lambda1 * d_lambda2
          + theta_d * (exp_lam1_m_lam2 / one_m_t_plus_t_prod_exp_lam1_m_lam2 - d_lambda1 * d_theta);
        d2_lambda2 
          = lambda2_d * (d_lambda2 - pow(d_lambda2,2.0)) 
          - lambda1_d * d_lambda1 * d_lambda2
          - theta_d * (1.0 / one_m_t_plus_t_prod_exp_lam1_m_lam2 + d_theta * d_lambda2);
        d2_theta_v 
          = lambda1_d * (exp_lam1_m_lam2_v / one_m_t_plus_t_prod_exp_lam1_m_lam2_v - d_lambda1_v * d_theta_v)
          - lambda2_d * (1.0 / one_m_t_plus_t_prod_exp_lam1_m_lam2_v + d_lambda2_v * d_theta_v)
          - theta_d * pow(d_theta_v,2.0);
        d2_lambda1_v 
          = lambda1_d * (d_lambda1_v - pow(d_lambda1_v,2.0))
          - lambda2_d * d_lambda1_v * d_lambda2_v
          + theta_d * (exp_lam1_m_lam2_v / one_m_t_plus_t_prod_exp_lam1_m_lam2_v - d_lambda1_v * d_theta_v);
        d2_lambda2_v 
          = lambda2_d * (d_lambda2_v - pow(d_lambda2_v,2.0)) 
          - lambda1_d * d_lambda1_v * d_lambda2_v
          - theta_d * (1.0 / one_m_t_plus_t_prod_exp_lam1_m_lam2_v + d_theta_v * d_lambda2_v);
      }

    double deriv 
      = d_theta * theta_d + d_lambda2 * lambda2_d 
               + d_lambda1 * lambda1_d;
    double deriv_2 
      = d2_theta * theta_d2 + d2_lambda1 * lambda1_d2
      + d2_lambda2 * lambda2_d2;
          
    var deriv_2_v 
      = d2_theta_v * theta_d2 + d2_lambda1_v * lambda1_d2
      + d2_lambda2_v * lambda2_d2;

    VEC d1_d2_d3;
    VEC d3 = cgrad(deriv_2_v,theta_v,lambda1_v,lambda2_v);

    d1_d2_d3.push_back(deriv);
    d1_d2_d3.push_back(deriv_2);
    d1_d2_d3.push_back(d_theta);
    d1_d2_d3.push_back(d_lambda1);
    d1_d2_d3.push_back(d_lambda2);
    d1_d2_d3.push_back(d2_theta);
    d1_d2_d3.push_back(d2_lambda1);
    d1_d2_d3.push_back(d2_lambda2);
    d1_d2_d3.push_back(d3[0]);
    d1_d2_d3.push_back(d3[1]);
    d1_d2_d3.push_back(d3[2]);

    stan::math::recover_memory();
    return d1_d2_d3;
} 

void test_log_mix_3xfvar_var_D2(double theta,
    double lambda1, double lambda2, double theta_d, 
    double lambda1_d, double lambda2_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<var> theta_fv(theta,theta_d);
  fvar<var> lambda1_fv(lambda1,lambda1_d);
  fvar<var> lambda2_fv(lambda2,lambda2_d);

  var theta_v(theta);
  var lambda1_v(lambda1);
  var lambda2_v(lambda2);

  var theta_d_v(theta_d);
  var lambda1_d_v(lambda1_d);
  var lambda2_d_v(lambda2_d);

  fvar<var> res = log_mix(theta_fv, lambda1_fv, lambda2_fv);
  double result = log_mix(theta_fv.val_.val(), lambda1_fv.val_.val(),
                          lambda2_fv.val_.val());

  double deriv_denom = exp(lambda1_fv.val_.val()) * theta_fv.val_.val()
    + exp(lambda2_fv.val_.val()) * (1 - theta_fv.val_.val());
  double theta_deriv = 1 / deriv_denom * (exp(lambda1_fv.val_.val()) 
        - exp(lambda2_fv.val_.val()));
  double lambda1_deriv = 1 / deriv_denom * exp(lambda1_fv.val_.val())
          * theta_fv.val_.val();
  double lambda2_deriv = 1 / deriv_denom * exp(lambda2_fv.val_.val()) 
    * (1 - theta_fv.val_.val());
  double deriv = theta_deriv * theta_fv.d_.val()
             + lambda2_deriv * lambda2_fv.d_.val() 
             + lambda1_deriv * lambda1_fv.d_.val();

  VEC g2_func = cgrad(res.d_,theta_fv.val_, lambda1_fv.val_,
     lambda2_fv.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             theta_d, lambda1_d, lambda2_d, 0, 0, 0);

  size_t k = 5;
  for (size_t i = 0; i < 3; ++i){
    EXPECT_NEAR(auto_calc[k],g2_func[i],8e-13) << "failed on " << k << std::endl;
    ++k;
  }

  EXPECT_FLOAT_EQ(res.d_.val(),auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_fvar_fvar_var_theta_D3(double theta,
    double lambda1, double lambda2, 
    double theta_d,double theta_d2){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<fvar<var> > theta_ffv;
  theta_ffv.val_.val_ = theta;
  theta_ffv.val_.d_ = theta_d2;
  theta_ffv.d_.val_ = theta_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1, lambda2);
  double result = log_mix(theta_ffv.val_.val_.val(), 
                          lambda1, 
                          lambda2);

  VEC g2_func = cgrad(res.d_.d_,theta_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             theta_d, 0, 0,
                             theta_d2, 0, 0);

  EXPECT_NEAR(auto_calc[8],g2_func[0],8e-13); 

  EXPECT_NEAR(auto_calc[1],res.d_.d_.val(),8e-13);
  EXPECT_FLOAT_EQ(res.d_.val_.val(),auto_calc[0]);
  EXPECT_NEAR(result, res.val_.val_.val(),8e-13);

  fvar<fvar<var> > theta_ffv_invalid;
  theta_ffv_invalid.val_.val_ = -1.0;
  theta_ffv_invalid.val_.d_ = theta_d2;
  theta_ffv_invalid.d_.val_ = theta_d;

  EXPECT_THROW(log_mix(theta_ffv_invalid,lambda1,lambda2),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(double theta,
    double lambda1, double lambda2, double lambda1_d, 
    double lambda1_d2){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<fvar<var> > lambda1_ffv;
  lambda1_ffv.val_.val_ = lambda1;
  lambda1_ffv.val_.d_ = lambda1_d2;
  lambda1_ffv.d_.val_ = lambda1_d;

  fvar<fvar<var> > res = log_mix(theta, lambda1_ffv, lambda2);
  double result = log_mix(theta, 
                          lambda1_ffv.val_.val_.val(), 
                          lambda2);

  VEC g2_func = cgrad(res.d_.d_,lambda1_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             0, lambda1_d, 0,
                             0, lambda1_d2, 0);

  EXPECT_NEAR(auto_calc[9],g2_func[0],8e-13); 

  EXPECT_NEAR(auto_calc[1],res.d_.d_.val(),8e-13);
  EXPECT_FLOAT_EQ(res.d_.val_.val(),auto_calc[0]);
  EXPECT_NEAR(result, res.val_.val_.val(),8e-13);

  EXPECT_THROW(log_mix(-1.0,lambda1_ffv,lambda2),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(double theta,
    double lambda1, double lambda2,double lambda2_d, 
    double lambda2_d2){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<fvar<var> > lambda2_ffv;
  lambda2_ffv.val_.val_ = lambda2;
  lambda2_ffv.val_.d_ = lambda2_d2;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta, lambda1, lambda2_ffv);
  double result = log_mix(theta, 
                          lambda1, 
                          lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.d_,lambda2_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             0, 0, lambda2_d,
                             0, 0, lambda2_d2);

  EXPECT_NEAR(auto_calc[10],g2_func[0],8e-13); 

  EXPECT_NEAR(auto_calc[1],res.d_.d_.val(),8e-13);
  EXPECT_FLOAT_EQ(res.d_.val_.val(),auto_calc[0]);
  EXPECT_NEAR(result, res.val_.val_.val(),8e-13);

  EXPECT_THROW(log_mix(-1.0,lambda1,lambda2_ffv),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_fvar_var_ex_theta_D3(double theta,
    double lambda1, double lambda2,  
    double lambda1_d, double lambda2_d,
    double lambda1_d2, double lambda2_d2){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<fvar<var> > lambda1_ffv;
  fvar<fvar<var> > lambda2_ffv;

  lambda1_ffv.val_.val_ = lambda1;
  lambda2_ffv.val_.val_ = lambda2;

  lambda1_ffv.val_.d_ = lambda1_d2;
  lambda2_ffv.val_.d_ = lambda2_d2;

  lambda1_ffv.d_.val_ = lambda1_d;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta, lambda1_ffv, lambda2_ffv);
  double result = log_mix(theta, 
                          lambda1_ffv.val_.val_.val(), 
                          lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.d_,lambda1_ffv.val_.val_,
     lambda2_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             0, lambda1_d, lambda2_d,
                             0, lambda1_d2, lambda2_d2);

  size_t k = 9;
  for (size_t i = 0; i < 2; ++i){
    EXPECT_NEAR(auto_calc[k],g2_func[i],8e-13) 
     << "failed on " << k << std::endl;
    ++k;
  }

  EXPECT_NEAR(auto_calc[1],res.d_.d_.val(),8e-13);
  EXPECT_FLOAT_EQ(res.d_.val_.val(),auto_calc[0]);
  EXPECT_NEAR(result, res.val_.val_.val(),8e-13);

  EXPECT_THROW(log_mix(-1.0,lambda1_ffv,lambda2_ffv),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(double theta,
    double lambda1, double lambda2, double theta_d, 
    double lambda2_d, double lambda2_d2, double theta_d2){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<fvar<var> > theta_ffv;
  fvar<fvar<var> > lambda2_ffv;

  theta_ffv.val_.val_ = theta;
  lambda2_ffv.val_.val_ = lambda2;

  theta_ffv.val_.d_ = theta_d2;
  lambda2_ffv.val_.d_ = lambda2_d2;

  theta_ffv.d_.val_ = theta_d;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1, lambda2_ffv);
  double result = log_mix(theta_ffv.val_.val_.val(), 
                          lambda1, 
                          lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.d_,theta_ffv.val_.val_,
     lambda2_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             theta_d, 0, lambda2_d,
                             theta_d2, 0, lambda2_d2);

  EXPECT_NEAR(auto_calc[8],g2_func[0],8e-13);
  EXPECT_NEAR(auto_calc[10],g2_func[1],8e-13);

  EXPECT_NEAR(auto_calc[1],res.d_.d_.val(),8e-13);
  EXPECT_FLOAT_EQ(res.d_.val_.val(),auto_calc[0]);
  EXPECT_NEAR(result, res.val_.val_.val(),8e-13);

  fvar<fvar<var> > theta_ffv_invalid;
  theta_ffv_invalid.val_.val_ = -1.0;
  theta_ffv_invalid.val_.d_ = theta_d2;
  theta_ffv_invalid.d_.val_ = theta_d;

  EXPECT_THROW(log_mix(theta_ffv_invalid,lambda1,lambda2_ffv),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(double theta,
    double lambda1, double lambda2, double theta_d, 
    double lambda1_d, double lambda1_d2, double theta_d2){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<fvar<var> > theta_ffv;
  fvar<fvar<var> > lambda1_ffv;

  theta_ffv.val_.val_ = theta;
  lambda1_ffv.val_.val_ = lambda1;

  theta_ffv.val_.d_ = theta_d2;
  lambda1_ffv.val_.d_ = lambda1_d2;

  theta_ffv.d_.val_ = theta_d;
  lambda1_ffv.d_.val_ = lambda1_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1_ffv, lambda2);
  double result = log_mix(theta_ffv.val_.val_.val(), 
                          lambda1_ffv.val_.val_.val(), 
                          lambda2);

  VEC g2_func = cgrad(res.d_.d_,theta_ffv.val_.val_, lambda1_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             theta_d, lambda1_d, 0,
                             theta_d2, lambda1_d2, 0);

  size_t k = 8;
  for (size_t i = 0; i < 2; ++i){
    EXPECT_NEAR(auto_calc[k],g2_func[i],8e-13) 
     << "failed on " << k << std::endl;
    ++k;
  }

  EXPECT_NEAR(auto_calc[1],res.d_.d_.val(),8e-13);
  EXPECT_FLOAT_EQ(res.d_.val_.val(),auto_calc[0]);
  EXPECT_NEAR(result, res.val_.val_.val(),8e-13);

  fvar<fvar<var> > theta_ffv_invalid;
  theta_ffv_invalid.val_.val_ = -1.0;
  theta_ffv_invalid.val_.d_ = theta_d2;
  theta_ffv_invalid.d_.val_ = theta_d;

  EXPECT_THROW(log_mix(theta_ffv_invalid,lambda1_ffv,lambda2),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_fvar_fvar_var_theta_D2(double theta,
    double lambda1, double lambda2, double theta_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<fvar<var> > theta_ffv;
  theta_ffv.val_.val_ = theta;
  theta_ffv.val_.d_ = theta_d;
  theta_ffv.d_.val_ = theta_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1, lambda2);
  double result = log_mix(theta_ffv.val_.val_.val(), 
                          lambda1, 
                          lambda2);

  VEC g2_func = cgrad(res.d_.val_,theta_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             theta_d, 0, 0,
                             0, 0, 0);

  EXPECT_NEAR(auto_calc[5],g2_func[0],8e-13);

  EXPECT_FLOAT_EQ(res.d_.val_.val(),auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val_.val());

  fvar<fvar<var> > theta_ffv_invalid;
  theta_ffv_invalid.val_.val_ = -1.0;
  theta_ffv_invalid.val_.d_ = theta_d;
  theta_ffv_invalid.d_.val_ = theta_d;

  EXPECT_THROW(log_mix(theta_ffv_invalid,lambda1,lambda2),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_fvar_fvar_var_lam_1_D2(double theta,
    double lambda1, double lambda2, double lambda1_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<fvar<var> > lambda1_ffv;
  lambda1_ffv.val_.val_ = lambda1;
  lambda1_ffv.val_.d_ = lambda1_d;
  lambda1_ffv.d_.val_ = lambda1_d;

  fvar<fvar<var> > res = log_mix(theta, lambda1_ffv, lambda2);
  double result = log_mix(theta, 
                          lambda1_ffv.val_.val_.val(), 
                          lambda2);

  VEC g2_func = cgrad(res.d_.val_,lambda1_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             0, lambda1_d, 0,
                             0, 0, 0);

  EXPECT_NEAR(auto_calc[6],g2_func[0],8e-13);

  EXPECT_FLOAT_EQ(res.d_.val_.val(),auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val_.val());

  EXPECT_THROW(log_mix(-1.0,lambda1_ffv,lambda2),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_fvar_fvar_var_lam_2_D2(double theta,
    double lambda1, double lambda2, double lambda2_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<fvar<var> > lambda2_ffv;
  lambda2_ffv.val_.val_ = lambda2;
  lambda2_ffv.val_.d_ = lambda2_d;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta, lambda1, lambda2_ffv);
  double result = log_mix(theta, 
                          lambda1, 
                          lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.val_,lambda2_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             0, 0, lambda2_d,
                             0, 0, 0);

  EXPECT_NEAR(auto_calc[7],g2_func[0],8e-13);

  EXPECT_FLOAT_EQ(res.d_.val_.val(),auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val_.val());

  EXPECT_THROW(log_mix(-1.0,lambda1,lambda2_ffv),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(double theta,
    double lambda1, double lambda2, double theta_d, 
    double lambda2_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<fvar<var> > theta_ffv;
  fvar<fvar<var> > lambda2_ffv;

  theta_ffv.val_.val_ = theta;
  lambda2_ffv.val_.val_ = lambda2;

  theta_ffv.val_.d_ = theta_d;
  lambda2_ffv.val_.d_ = lambda2_d;

  theta_ffv.d_.val_ = theta_d;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1, lambda2_ffv);
  double result = log_mix(theta_ffv.val_.val_.val(), 
                          lambda1, 
                          lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.val_,theta_ffv.val_.val_,lambda2_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             theta_d, 0, lambda2_d,
                             0, 0, 0);

  EXPECT_NEAR(auto_calc[5],g2_func[0],8e-13); 
  EXPECT_NEAR(auto_calc[7],g2_func[1],8e-13);

  EXPECT_FLOAT_EQ(res.d_.val_.val(),auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val_.val());

  fvar<fvar<var> > theta_ffv_invalid;
  theta_ffv_invalid.val_.val_ = -1.0;
  theta_ffv_invalid.val_.d_ = theta_d;
  theta_ffv_invalid.d_.val_ = theta_d;

  EXPECT_THROW(log_mix(theta_ffv_invalid,lambda1,lambda2_ffv),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(double theta,
    double lambda1, double lambda2, double theta_d, 
    double lambda1_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<fvar<var> > theta_ffv;
  fvar<fvar<var> > lambda1_ffv;

  theta_ffv.val_.val_ = theta;
  lambda1_ffv.val_.val_ = lambda1;

  theta_ffv.val_.d_ = theta_d;
  lambda1_ffv.val_.d_ = lambda1_d;

  theta_ffv.d_.val_ = theta_d;
  lambda1_ffv.d_.val_ = lambda1_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1_ffv, lambda2);
  double result = log_mix(theta_ffv.val_.val_.val(), 
                          lambda1_ffv.val_.val_.val(), 
                          lambda2);

  VEC g2_func = cgrad(res.d_.val_,theta_ffv.val_.val_, lambda1_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             theta_d, lambda1_d, 0,
                             0, 0, 0);

  size_t k = 5;
  for (size_t i = 0; i < 2; ++i){
    EXPECT_NEAR(auto_calc[k],g2_func[i],8e-13) << "failed on " << k << std::endl;
    ++k;
  }

  EXPECT_FLOAT_EQ(res.d_.val_.val(),auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val_.val());

  fvar<fvar<var> > theta_ffv_invalid;
  theta_ffv_invalid.val_.val_ = -1.0;
  theta_ffv_invalid.val_.d_ = theta_d;
  theta_ffv_invalid.d_.val_ = theta_d;

  EXPECT_THROW(log_mix(theta_ffv_invalid,lambda1_ffv,lambda2),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_fvar_var_ex_theta_D2(double theta,
    double lambda1, double lambda2, 
    double lambda1_d, double lambda2_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<fvar<var> > lambda1_ffv;
  fvar<fvar<var> > lambda2_ffv;

  lambda1_ffv.val_.val_ = lambda1;
  lambda2_ffv.val_.val_ = lambda2;

  lambda1_ffv.val_.d_ = lambda1_d;
  lambda2_ffv.val_.d_ = lambda2_d;

  lambda1_ffv.d_.val_ = lambda1_d;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta, lambda1_ffv, lambda2_ffv);
  double result = log_mix(theta, 
                          lambda1_ffv.val_.val_.val(), 
                          lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.val_,lambda1_ffv.val_.val_,
     lambda2_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             0, lambda1_d, lambda2_d,
                             0, 0, 0);

  size_t k = 6;
  for (size_t i = 0; i < 2; ++i){
    EXPECT_NEAR(auto_calc[k],g2_func[i],8e-13) << "failed on " << k << std::endl;
    ++k;
  }

  EXPECT_FLOAT_EQ(res.d_.val_.val(),auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val_.val());

  EXPECT_THROW(log_mix(-1.0,lambda1_ffv,lambda2_ffv),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_3xfvar_fvar_var_D3(double theta,
    double lambda1, double lambda2, double theta_d, 
    double lambda1_d, double lambda2_d,
    double lambda1_d2, double lambda2_d2, double theta_d2){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<fvar<var> > theta_ffv;
  fvar<fvar<var> > lambda1_ffv;
  fvar<fvar<var> > lambda2_ffv;

  theta_ffv.val_.val_ = theta;
  lambda1_ffv.val_.val_ = lambda1;
  lambda2_ffv.val_.val_ = lambda2;

  theta_ffv.val_.d_ = theta_d2;
  lambda1_ffv.val_.d_ = lambda1_d2;
  lambda2_ffv.val_.d_ = lambda2_d2;

  theta_ffv.d_.val_ = theta_d;
  lambda1_ffv.d_.val_ = lambda1_d;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1_ffv, lambda2_ffv);
  double result = log_mix(theta_ffv.val_.val_.val(), 
                          lambda1_ffv.val_.val_.val(), 
                          lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.d_,theta_ffv.val_.val_, lambda1_ffv.val_.val_,
     lambda2_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             theta_d, lambda1_d, lambda2_d,
                             theta_d2, lambda1_d2, lambda2_d2);

  size_t k = 8;
  for (size_t i = 0; i < 3; ++i){
    EXPECT_NEAR(auto_calc[k],g2_func[i],8e-13) 
     << "failed on " << k << std::endl;
    ++k;
  }

  EXPECT_NEAR(auto_calc[1],res.d_.d_.val(),8e-13);
  EXPECT_FLOAT_EQ(res.d_.val_.val(),auto_calc[0]);
  EXPECT_NEAR(result, res.val_.val_.val(),8e-13);

  fvar<fvar<var> > theta_ffv_invalid;
  theta_ffv_invalid.val_.val_ = -1.0;
  theta_ffv_invalid.val_.d_ = theta_d2;
  theta_ffv_invalid.d_.val_ = theta_d;

  EXPECT_THROW(log_mix(theta_ffv_invalid,lambda1_ffv,lambda2_ffv),std::domain_error);
  stan::math::recover_memory();
}


void test_log_mix_3xfvar_fvar_var_D2(double theta,
    double lambda1, double lambda2, double theta_d, 
    double lambda1_d, double lambda2_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using std::pow;
  using stan::math::log_mix;

  fvar<fvar<var> > theta_ffv;
  fvar<fvar<var> > lambda1_ffv;
  fvar<fvar<var> > lambda2_ffv;

  theta_ffv.val_.val_ = theta;
  lambda1_ffv.val_.val_ = lambda1;
  lambda2_ffv.val_.val_ = lambda2;

  theta_ffv.val_.d_ = theta_d;
  lambda1_ffv.val_.d_ = lambda1_d;
  lambda2_ffv.val_.d_ = lambda2_d;

  theta_ffv.d_.val_ = theta_d;
  lambda1_ffv.d_.val_ = lambda1_d;
  lambda2_ffv.d_.val_ = lambda2_d;

  fvar<fvar<var> > res = log_mix(theta_ffv, lambda1_ffv, lambda2_ffv);
  double result = log_mix(theta_ffv.val_.val_.val(), 
                          lambda1_ffv.val_.val_.val(), 
                          lambda2_ffv.val_.val_.val());

  VEC g2_func = cgrad(res.d_.val_,theta_ffv.val_.val_, lambda1_ffv.val_.val_,
     lambda2_ffv.val_.val_);

  VEC auto_calc = log_mix_D3(theta, lambda1, lambda2,
                             theta_d, lambda1_d, lambda2_d,
                             0, 0, 0);

  size_t k = 5;
  for (size_t i = 0; i < 3; ++i){
    EXPECT_NEAR(auto_calc[k],g2_func[i],8e-13) << "failed on " << k << std::endl;
    ++k;
  }

  EXPECT_FLOAT_EQ(res.d_.val_.val(),auto_calc[0]);
  EXPECT_FLOAT_EQ(result, res.val_.val_.val());
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_var_lam_2_double(double theta,
    double lambda1, double lambda2, double theta_d, 
    double lambda1_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using stan::math::log_mix;

  fvar<var> theta_fv(theta,theta_d);
  fvar<var> lambda1_fv(lambda1,lambda1_d);

  fvar<var> res = log_mix(theta_fv, lambda1_fv, lambda2);
  double result = log_mix(theta_fv.val_.val(), lambda1_fv.val_.val(),
                          lambda2);
  double deriv_denom = exp(lambda1_fv.val_.val()) * theta_fv.val_.val()
    + exp(lambda2) * (1 - theta_fv.val_.val());
  double theta_deriv = 1 / deriv_denom * (exp(lambda1_fv.val_.val()) 
        - exp(lambda2));
  double lambda1_deriv = 1 / deriv_denom * exp(lambda1_fv.val_.val())
          * theta_fv.val_.val();
  double deriv = theta_deriv * theta_fv.d_.val()
        + lambda1_deriv * lambda1_fv.d_.val();

  AVEC y = createAVEC(theta_fv.val_, lambda1_fv.val_);
  VEC g; 
  res.val_.grad(y,g);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(theta_deriv,g[0]);
  EXPECT_FLOAT_EQ(lambda1_deriv,g[1]);

  fvar<var> theta_fv_invalid(-1.0,theta_d);

  EXPECT_THROW(log_mix(theta_fv_invalid,lambda1_fv,lambda2),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_var_lam_1_double(double theta,
            double lambda1, double lambda2, double theta_d, 
            double lambda2_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using stan::math::log_mix;

  fvar<var> theta_fv(theta,theta_d);
  fvar<var> lambda2_fv(lambda2,lambda2_d);

  fvar<var> res = log_mix(theta_fv, lambda1, lambda2_fv);
  double result = log_mix(theta_fv.val_.val(), lambda1,
                          lambda2_fv.val_.val());
  double deriv_denom = exp(lambda1) * theta_fv.val_.val()
    + exp(lambda2_fv.val_.val()) * (1 - theta_fv.val_.val());
  double theta_deriv = 1 / deriv_denom * (exp(lambda1) 
        - exp(lambda2_fv.val_.val()));
  double lambda2_deriv = 1 / deriv_denom * exp(lambda2_fv.val_.val()) 
    * (1 - theta_fv.val_.val());
  double deriv = theta_deriv * theta_fv.d_.val()
        + lambda2_deriv * lambda2_fv.d_.val(); 

  AVEC y = createAVEC(theta_fv.val_,
     lambda2_fv.val_);
  VEC g; 
  res.val_.grad(y,g);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(theta_deriv,g[0]);
  EXPECT_FLOAT_EQ(lambda2_deriv,g[1]);

  fvar<var> theta_fv_invalid(-1.0,theta_d);

  EXPECT_THROW(log_mix(theta_fv_invalid,lambda1,lambda2_fv),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xfvar_var_theta_double(double theta,
    double lambda1, double lambda2, 
    double lambda1_d, double lambda2_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using stan::math::log_mix;

  fvar<var> lambda1_fv(lambda1,lambda1_d);
  fvar<var> lambda2_fv(lambda2,lambda2_d);

  fvar<var> res = log_mix(theta, lambda1_fv, lambda2_fv);
  double result = log_mix(theta, lambda1_fv.val_.val(),
                          lambda2_fv.val_.val());
  double deriv_denom = exp(lambda1_fv.val_.val()) * theta
    + exp(lambda2_fv.val_.val()) * (1 - theta);
  double lambda1_deriv = 1 / deriv_denom * exp(lambda1_fv.val_.val())
          * theta;
  double lambda2_deriv = 1 / deriv_denom * exp(lambda2_fv.val_.val()) 
    * (1 - theta);
  double deriv = lambda2_deriv * lambda2_fv.d_.val() 
        + lambda1_deriv * lambda1_fv.d_.val();

  AVEC y = createAVEC(lambda1_fv.val_,
     lambda2_fv.val_);
  VEC g; 
  res.val_.grad(y,g);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(lambda1_deriv,g[0]);
  EXPECT_FLOAT_EQ(lambda2_deriv,g[1]);

  EXPECT_THROW(log_mix(-1.0,lambda1_fv,lambda2_fv),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_theta_fvar_var(double theta,
    double lambda1, double lambda2, double theta_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using stan::math::log_mix;

  fvar<var> theta_fv(theta,theta_d);

  fvar<var> res = log_mix(theta_fv, lambda1, lambda2);
  double result = log_mix(theta_fv.val_.val(), lambda1,
                          lambda2);
  double deriv_denom = exp(lambda1) * theta_fv.val_.val()
    + exp(lambda2) * (1 - theta_fv.val_.val());
  double theta_deriv = 1 / deriv_denom * (exp(lambda1) 
        - exp(lambda2));
  double deriv = theta_deriv * theta_fv.d_.val();

  AVEC y = createAVEC(theta_fv.val_);
  VEC g; 
  res.val_.grad(y,g);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(theta_deriv,g[0]);

  fvar<var> theta_fv_invalid(-1.0,theta_d);

  EXPECT_THROW(log_mix(theta_fv_invalid,lambda1,lambda2),std::domain_error);
  stan::math::recover_memory();
}

void test_log_mix_2xdouble_lam_1_fvar_var(double theta,
    double lambda1, double lambda2, 
    double lambda1_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using stan::math::log_mix;

  fvar<var> lambda1_fv(lambda1,lambda1_d);

  fvar<var> res = log_mix(theta, lambda1_fv, lambda2);
  double result = log_mix(theta, lambda1_fv.val_.val(),
                          lambda2);
  double deriv_denom = exp(lambda1_fv.val_.val()) * theta
    + exp(lambda2) * (1 - theta);
  double lambda1_deriv = 1 / deriv_denom * exp(lambda1_fv.val_.val())
          * theta;
  double deriv = lambda1_deriv * lambda1_fv.d_.val();

  AVEC y = createAVEC(lambda1_fv.val_);
  VEC g; 
  res.val_.grad(y,g);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(lambda1_deriv,g[0]);

  EXPECT_THROW(log_mix(-1.0,lambda1_fv,lambda2),std::domain_error);
  stan::math::recover_memory();
}


void test_log_mix_2xdouble_lam_2_fvar_var(double theta,
    double lambda1, double lambda2, double lambda2_d){
  using stan::math::var;
  using stan::math::fvar;
  using std::exp;
  using stan::math::log_mix;

  fvar<var> lambda2_fv(lambda2,lambda2_d);

  fvar<var> res = log_mix(theta, lambda1, lambda2_fv);
  double result = log_mix(theta, lambda1,
                          lambda2_fv.val_.val());
  double deriv_denom = exp(lambda1) * theta
    + exp(lambda2_fv.val_.val()) * (1 - theta);
  double lambda2_deriv = 1 / deriv_denom * exp(lambda2_fv.val_.val()) 
    * (1 - theta);
  double deriv = lambda2_deriv * lambda2_fv.d_.val(); 

  AVEC y = createAVEC(lambda2_fv.val_);
  VEC g; 
  res.val_.grad(y,g);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(lambda2_deriv,g[0]);

  EXPECT_THROW(log_mix(-1.0,lambda1,lambda2_fv),std::domain_error);
  stan::math::recover_memory();
}

TEST(AgradFwdLogMix, FvarFvarVar_Double_Double_D3){

  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.7, 2.0, 6.0, 1.3, 5.0);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.7, 2.0, 6.0, 1, 0);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.3, 2.0, 6.0, 0, 0);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.3, 1.0, 2.0, 0, 1);

  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.7, 2.0, -6.0, 1.3, 5.0);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.7, 2.0, -6.0, 1, 0);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.3, 2.0, -6.0, 0, 0);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D3(0.3, 1.0, -2.0, 0, 1);

  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.7, 2.0, 6.0, 1.3, 5.0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.7, 2.0, 6.0, 1, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.3, 2.0, 6.0, 0, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.3, 1.0, 2.0, 0, 1);

  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.7, 2.0, -6.0, 1.3, 5.0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.7, 2.0, -6.0, 1, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.3, 2.0, -6.0, 0, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D3(0.3, 1.0, -2.0, 0, 1);

  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.7, 2.0, 6.0, 1.3, 5.0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.7, 2.0, 6.0, 1, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.3, 2.0, 6.0, 0, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.3, 1.0, 2.0, 0, 1);

  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.7, 2.0, -6.0, 1.3, 5.0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.7, 2.0, -6.0, 1, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.3, 2.0, -6.0, 0, 0);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D3(0.3, 1.0, -2.0, 0, 1);
}

TEST(AgradFwdLogMix, FvarFvarVar_Double_Double_D2){

  test_log_mix_2xdouble_fvar_fvar_var_theta_D2(0.7, 2.0, 6.0, 1.3);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D2(0.7, 2.0, 6.0, 1);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D2(0.7, 2.0, 6.0, 0);

  test_log_mix_2xdouble_fvar_fvar_var_theta_D2(0.3, 2.0, -6.0, 1.3);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D2(0.3, 2.0, -6.0, 1);
  test_log_mix_2xdouble_fvar_fvar_var_theta_D2(0.3, 2.0, -6.0, 0);

  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D2(0.7, 2.0, 6.0, 1.3);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D2(0.7, 2.0, 6.0, 1);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D2(0.7, 2.0, 6.0, 0);

  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D2(0.3, 2.0, -6.0, 1.3);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D2(0.3, 2.0, -6.0, 1);
  test_log_mix_2xdouble_fvar_fvar_var_lam_1_D2(0.3, 2.0, -6.0, 0);

  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D2(0.7, 2.0, 6.0, 1.3);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D2(0.7, 2.0, 6.0, 1);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D2(0.7, 2.0, 6.0, 0);

  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D2(0.3, 2.0, -6.0, 1.3);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D2(0.3, 2.0, -6.0, 1);
  test_log_mix_2xdouble_fvar_fvar_var_lam_2_D2(0.3, 2.0, -6.0, 0);
}

TEST(AgradFwdLogMix, FvarVar_FvarVar_FvarVar_D1){

  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 1.3, 5.0, 3.0);
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 1, 0, 0);
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 0, 1, 0);
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 0, 0, 1);
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 1, 0, 1);
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 0, 1, 1);
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 1, 1, 0);
  test_log_mix_3xfvar_var_D1(0.7, 2.0, 6.0, 1, 1, 1);

  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 1.3, 2.0, 3.0);
  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 1, 0, 0);
  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 0, 1, 0);
  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 0, 0, 1);
  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 1, 0, 1);
  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 0, 1, 1);
  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 1, 1, 0);
  test_log_mix_3xfvar_var_D1(0.2, 2.0, -6.0, 1, 1, 1);
}

TEST(AgradFwdLogMix, FvarFvarVar_FvarFvarVar_Double_D3){

  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, 6.0, 5.0, 3.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, 6.0, 1, 1, 1, 1);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, 6.0, 1, 1, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, 6.0, 1, 0, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, 6.0, 1, 0, 1, 1);

  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, -6.0, 5.0, 3.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, -6.0, 1, 1, 1, 1);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, -6.0, 1, 1, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, -6.0, 1, 0, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_theta_D3(0.7, 2.0, -6.0, 1, 0, 1, 1);

  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, 6.0, 5.0, 3.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, 6.0, 1, 1, 1, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, 6.0, 1, 1, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, 6.0, 1, 0, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, 6.0, 1, 0, 1, 1);

  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, -6.0, 5.0, 3.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, -6.0, 1, 1, 1, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, -6.0, 1, 1, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, -6.0, 1, 0, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D3(0.7, 2.0, -6.0, 1, 0, 1, 1); 

  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, 6.0, 5.0, 3.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, 6.0, 1, 1, 1, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, 6.0, 1, 1, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, 6.0, 1, 0, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, 6.0, 1, 0, 1, 1);

  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, -6.0, 5.0, 3.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, -6.0, 1, 1, 1, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, -6.0, 1, 1, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, -6.0, 1, 0, 0, 1);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D3(0.7, 2.0, -6.0, 1, 0, 1, 1);
}

TEST(AgradFwdLogMix, FvarFvarVar_FvarFvarVar_Double_D2){

  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, 6.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, 6.0, 1.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, 6.0, 1.0, 1.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, 6.0, 0.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, 6.0, 0.0, 1.0);

  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, -6.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, -6.0, 1.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, -6.0, 1.0, 1.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, -6.0, 0.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_theta_D2(0.7, 2.0, -6.0, 0.0, 1.0);

  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, 6.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, 6.0, 1.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, 6.0, 1.0, 1.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, 6.0, 0.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, 6.0, 0.0, 1.0);

  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, -6.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, -6.0, 1.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, -6.0, 1.0, 1.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, -6.0, 0.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_1_D2(0.7, 2.0, -6.0, 0.0, 1.0);

  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, 6.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, 6.0, 1.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, 6.0, 1.0, 1.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, 6.0, 0.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, 6.0, 0.0, 1.0);

  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, -6.0, 5.0, 3.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, -6.0, 1.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, -6.0, 1.0, 1.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, -6.0, 0.0, 0.0);
  test_log_mix_2xfvar_fvar_var_ex_lam_2_D2(0.7, 2.0, -6.0, 0.0, 1.0);
}

TEST(AgradFwdLogMix, FvarVar_FvarVar_FvarVar_D2){

  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 1.3, 5.0, 3.0);
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 1.0, 0.0, 0);
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 0.0, 1.0, 0.0);
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 0.0, 0.0, 1.0);
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 1.0, 0.0, 1.0);
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 0.0, 1.0, 1.0); //
  test_log_mix_3xfvar_var_D2(0.7, 4.0, 5.0, 0.0, 1.0, 1.0); //
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 1.0, 1.0, 0.0);
  test_log_mix_3xfvar_var_D2(0.7, 2.0, 6.0, 1.0, 1.0, 1.0);

  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 1.3, 2.0, 3.0);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 1.0, 0.0, 0.0);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.0, 0.0);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 0.0, 0.0, 1.0);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 1.0, 0.0, 1.0);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.0, 1.0); //
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.1, 1.0); 
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.1, 1.1); //
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.0, 1.1); 
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 1.0, 1.0, 0.0);
  test_log_mix_3xfvar_var_D2(0.2, 2.0, -6.0, 1.0, 1.0, 1.0);
}

TEST(AgradFwdLogMix, FvarFvarVar_FvarFvarVar_FvarFvarVar_D2){

  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 1.3, 5.0, 3.0);
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 1.0, 0.0, 0);
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 0.0, 1.0, 0.0);
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 0.0, 0.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 1.0, 0.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 0.0, 1.0, 1.0); //
  test_log_mix_3xfvar_fvar_var_D2(0.7, 4.0, 5.0, 0.0, 1.0, 1.0); //
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 1.0, 1.0, 0.0);
  test_log_mix_3xfvar_fvar_var_D2(0.7, 2.0, 6.0, 1.0, 1.0, 1.0);

  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 1.3, 2.0, 3.0);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 1.0, 0.0, 0.0);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.0, 0.0);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 0.0, 0.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 1.0, 0.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.0, 1.0); //
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.1, 1.0); 
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.1, 1.1); //
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 0.0, 1.0, 1.1); 
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 1.0, 1.0, 0.0);
  test_log_mix_3xfvar_fvar_var_D2(0.2, 2.0, -6.0, 1.0, 1.0, 1.0); 
}

TEST(AgradFwdLogMix, FvarFvarVar_FvarFvarVar_FvarFvarVar_D3){

  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 1.3, 5.0, 3.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 1.0, 0.0, 0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 0.0, 1.0, 0.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 0.0, 0.0, 1.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 1.0, 0.0, 1.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 0.0, 1.0, 1.0, 3.0, 4.0, 1.0); //
  test_log_mix_3xfvar_fvar_var_D3(0.7, 4.0, 5.0, 0.0, 1.0, 1.0, 3.0, 4.0, 1.0); //
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 1.0, 1.0, 0.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.7, 2.0, 6.0, 1.0, 1.0, 1.0, 3.0, 4.0, 1.0);

  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 1.3, 2.0, 3.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 1.0, 0.0, 0.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 0.0, 1.0, 0.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 0.0, 0.0, 1.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 1.0, 0.0, 1.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 0.0, 1.0, 1.0, 3.0, 4.0, 1.0); //
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 0.0, 1.1, 1.0, 3.0, 4.0, 1.0); 
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 0.0, 1.1, 1.1, 3.0, 4.0, 1.0); //
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 0.0, 1.0, 1.1, 3.0, 4.0, 1.0); 
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 1.0, 1.0, 0.0, 3.0, 4.0, 1.0);
  test_log_mix_3xfvar_fvar_var_D3(0.2, 2.0, -6.0, 1.0, 1.0, 1.0, 3.0, 4.0, 1.0); 
}

TEST(AgradFwdLogMix, FvarVar_FvarVar_Double){

  test_log_mix_2xfvar_var_lam_1_double(0.7, 2.0, 6.0, 1.3, 2.0);
  test_log_mix_2xfvar_var_lam_1_double(0.7, 2.0, 6.0, 1, 0);
  test_log_mix_2xfvar_var_lam_1_double(0.7, 2.0, 6.0, 0, 1);
  test_log_mix_2xfvar_var_lam_1_double(0.7, 2.0, 6.0, 1, 1);
  test_log_mix_2xfvar_var_lam_1_double(0.7, 2.0, 6.0, 0, 0);

  test_log_mix_2xfvar_var_lam_1_double(0.2, -2.0, 6.0, 1, 0);
  test_log_mix_2xfvar_var_lam_1_double(0.2, -2.0, 6.0, 0, 1);
  test_log_mix_2xfvar_var_lam_1_double(0.2, -2.0, 6.0, 1, 1);
  test_log_mix_2xfvar_var_lam_1_double(0.2, -2.0, 6.0, 0, 0);

  test_log_mix_2xfvar_var_lam_2_double(0.7, 2.0, 6.0, 1.3, 2.0);
  test_log_mix_2xfvar_var_lam_2_double(0.7, 2.0, 6.0, 1, 0);
  test_log_mix_2xfvar_var_lam_2_double(0.7, 2.0, 6.0, 0, 1);
  test_log_mix_2xfvar_var_lam_2_double(0.7, 2.0, 6.0, 1, 1);
  test_log_mix_2xfvar_var_lam_2_double(0.7, 2.0, 6.0, 0, 0);

  test_log_mix_2xfvar_var_lam_2_double(0.2, -2.0, 6.0, 1, 0);
  test_log_mix_2xfvar_var_lam_2_double(0.2, -2.0, 6.0, 0, 1);
  test_log_mix_2xfvar_var_lam_2_double(0.2, -2.0, 6.0, 1, 1);
  test_log_mix_2xfvar_var_lam_2_double(0.2, -2.0, 6.0, 0, 0);

  test_log_mix_2xfvar_var_theta_double(0.7, 2.0, 6.0, 1.3, 2.0);
  test_log_mix_2xfvar_var_theta_double(0.7, 2.0, 6.0, 1, 0);
  test_log_mix_2xfvar_var_theta_double(0.7, 2.0, 6.0, 0, 1);
  test_log_mix_2xfvar_var_theta_double(0.7, 2.0, 6.0, 1, 1);
  test_log_mix_2xfvar_var_theta_double(0.7, 2.0, 6.0, 0, 0);

  test_log_mix_2xfvar_var_theta_double(0.2, -2.0, 6.0, 1, 0);
  test_log_mix_2xfvar_var_theta_double(0.2, -2.0, 6.0, 0, 1);
  test_log_mix_2xfvar_var_theta_double(0.2, -2.0, 6.0, 1, 1);
  test_log_mix_2xfvar_var_theta_double(0.2, -2.0, 6.0, 0, 0);
}

TEST(AgradFwdLogMix, FvarVar_Double_Double){

  test_log_mix_2xdouble_theta_fvar_var(0.7, 2.0, 6.0, 1.3);
  test_log_mix_2xdouble_theta_fvar_var(0.7, 2.0, 6.0, 1);
  test_log_mix_2xdouble_theta_fvar_var(0.7, 2.0, 6.0, 0);

  test_log_mix_2xdouble_theta_fvar_var(0.2, -2.0, 6.0, 1);
  test_log_mix_2xdouble_theta_fvar_var(0.2, -2.0, 6.0, 0);

  test_log_mix_2xdouble_lam_1_fvar_var(0.7, 2.0, 6.0, 0.3);
  test_log_mix_2xdouble_lam_1_fvar_var(0.7, 2.0, 6.0, 1);
  test_log_mix_2xdouble_lam_1_fvar_var(0.7, 2.0, 6.0, 0);

  test_log_mix_2xdouble_lam_1_fvar_var(0.2, -2.0, 6.0, 1);
  test_log_mix_2xdouble_lam_1_fvar_var(0.2, -2.0, 6.0, 0);

  test_log_mix_2xdouble_lam_2_fvar_var(0.7, 2.0, 6.0, 1.3);
  test_log_mix_2xdouble_lam_2_fvar_var(0.7, 2.0, 6.0, 1);
  test_log_mix_2xdouble_lam_2_fvar_var(0.7, 2.0, 6.0, 0);

  test_log_mix_2xdouble_lam_2_fvar_var(0.2, -2.0, 6.0, 1);
  test_log_mix_2xdouble_lam_2_fvar_var(0.2, -2.0, 6.0, 0); 
} 

struct log_mix_fun {
  template <typename T0, typename T1, typename T2>
  inline 
  typename boost::math::tools::promote_args<T0,T1,T2>::type
  operator()(const T0 arg1,
             const T1 arg2,
             const T2 arg3) const {
    return log_mix(arg1,arg2,arg3);
  }
};

TEST(AgradFwdLogMix, log_mix_NaN) {
  log_mix_fun log_mix_;
  test_nan_mix(log_mix_,0.7,3.0,5.0,true);
}
