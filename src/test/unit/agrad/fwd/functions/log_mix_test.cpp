#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/math/functions/log_mix.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <cmath>


void test_log_mix_fff(double theta, double lambda1, double lambda2,
                      double theta_d, double lambda1_d, double lambda2_d){
  using stan::agrad::fvar;
  using stan::math::log_mix;

  fvar<double> theta_f(theta,theta_d);
  fvar<double> lambda1_f(lambda1,lambda1_d);
  fvar<double> lambda2_f(lambda2,lambda2_d);

  fvar<double> f = log_mix(theta_f, lambda1_f, lambda2_f);
  fvar<double> f2 = log(theta_f * exp(lambda1_f) + (1 - theta_f) * exp(lambda2_f));
  EXPECT_FLOAT_EQ(f.val_,f2.val_);
  EXPECT_FLOAT_EQ(f.d_,f2.d_);
}

void test_log_mix_f_explicit(double theta, double lambda1, double x){
  using stan::agrad::fvar;
  using stan::math::log_mix;
  using std::exp;
  using ::cos;
  using ::sin;

  fvar<double> x_f(x,1);
  fvar<double> lambda2_f = sin(x_f);

  fvar<double> f = log_mix(theta, lambda1, lambda2_f);
  fvar<double> f2 = log(theta * exp(lambda1) + (1 - theta) * exp(sin(x_f)));
  double num_deriv = exp(sin(x_f.val_)) * (1 - theta) 
    * cos(x_f.val_) / (exp(sin(x_f.val_)) * (1 - theta) 
            + exp(lambda1) * theta);
  EXPECT_FLOAT_EQ(f.val_,f2.val_);
  EXPECT_FLOAT_EQ(f.d_,f2.d_);
  EXPECT_FLOAT_EQ(f.d_,num_deriv);
}

void test_log_mix_ff_ex_lam_2(double theta, double lambda1, double lambda2,
                              double theta_d, double lambda1_d){
  using stan::agrad::fvar;
  using stan::math::log_mix;

  fvar<double> theta_f(theta,theta_d);
  fvar<double> lambda1_f(lambda1,lambda1_d);

  fvar<double> f = log_mix(theta_f, lambda1_f, lambda2);
  fvar<double> f2 = log(theta_f * exp(lambda1_f) + (1 - theta_f) * exp(lambda2));
  EXPECT_FLOAT_EQ(f.val_,f2.val_);
  EXPECT_FLOAT_EQ(f.d_,f2.d_);
}

void test_log_mix_ff_ex_lam_1(double theta, double lambda1, double lambda2,
                              double theta_d, double lambda2_d){
  using stan::agrad::fvar;
  using stan::math::log_mix;

  fvar<double> theta_f(theta,theta_d);
  fvar<double> lambda2_f(lambda2,lambda2_d);

  fvar<double> f = log_mix(theta_f, lambda1, lambda2_f);
  fvar<double> f2 = log(theta_f * exp(lambda1) + (1 - theta_f) * exp(lambda2_f));
  EXPECT_FLOAT_EQ(f.val_,f2.val_);
  EXPECT_FLOAT_EQ(f.d_,f2.d_);
}

void test_log_mix_ff_ex_theta(double theta, double lambda1, double lambda2,
                              double lambda1_d, double lambda2_d){
  using stan::agrad::fvar;
  using stan::math::log_mix;

  fvar<double> lambda1_f(lambda1,lambda1_d);
  fvar<double> lambda2_f(lambda2,lambda2_d);

  fvar<double> f = log_mix(theta, lambda1_f, lambda2_f);
  fvar<double> f2 = log(theta * exp(lambda1_f) + (1 - theta) * exp(lambda2_f));
  EXPECT_FLOAT_EQ(f.val_,f2.val_);
  EXPECT_FLOAT_EQ(f.d_,f2.d_);
}

TEST(AgradFwdLogMix,Fvar) {
  test_log_mix_fff(0.7, 1.5, -2.0, 1, 0, 0);
  test_log_mix_fff(0.7, 1.5, -2.0, 0, 1, 0);
  test_log_mix_fff(0.7, 1.5, -2.0, 0, 0, 1);

  test_log_mix_fff(0.7, -1.5, 2.0, 1, 0, 0);
  test_log_mix_fff(0.7, -1.5, 2.0, 0, 1, 0);
  test_log_mix_fff(0.7, -1.5, 2.0, 0, 0, 1);

  test_log_mix_ff_ex_lam_2(0.7, 1.5, -2.0, 1, 0);
  test_log_mix_ff_ex_lam_2(0.7, 1.5, -2.0, 0, 1);
  test_log_mix_ff_ex_lam_2(0.7, 1.5, -2.0, 0, 0);

  test_log_mix_ff_ex_lam_2(0.7, -1.5, 2.0, 1, 0);
  test_log_mix_ff_ex_lam_2(0.7, -1.5, 2.0, 0, 1);
  test_log_mix_ff_ex_lam_2(0.7, -1.5, 2.0, 0, 0);

  test_log_mix_ff_ex_lam_1(0.7, 1.5, -2.0, 1, 0);
  test_log_mix_ff_ex_lam_1(0.7, 1.5, -2.0, 0, 1);
  test_log_mix_ff_ex_lam_1(0.7, 1.5, -2.0, 0, 0);

  test_log_mix_ff_ex_lam_1(0.7, -1.5, 2.0, 1, 0);
  test_log_mix_ff_ex_lam_1(0.7, -1.5, 2.0, 0, 1);
  test_log_mix_ff_ex_lam_1(0.7, -1.5, 2.0, 0, 0);

  test_log_mix_ff_ex_theta(0.7, 1.5, -2.0, 1, 0);
  test_log_mix_ff_ex_theta(0.7, 1.5, -2.0, 0, 1);
  test_log_mix_ff_ex_theta(0.7, 1.5, -2.0, 0, 0);

  test_log_mix_ff_ex_theta(0.7, -1.5, 2.0, 1, 0);
  test_log_mix_ff_ex_theta(0.7, -1.5, 2.0, 0, 1);
  test_log_mix_ff_ex_theta(0.7, -1.5, 2.0, 0, 0);

  test_log_mix_f_explicit(0.7, 1.5, 5);
  test_log_mix_f_explicit(0.7, 0.1, 5);
  test_log_mix_f_explicit(0.999, 0.1, 5);
  test_log_mix_f_explicit(0.0001, 0.1, 5);
}  

void test_log_mix_3xfvar_var(double theta,
    double lambda1, double lambda2, double theta_d, 
    double lambda1_d, double lambda2_d){
  using stan::agrad::var;
  using stan::agrad::fvar;
  using std::exp;
  using stan::math::log_mix;

  fvar<var> theta_v(theta,theta_d);
  fvar<var> lambda1_v(lambda1,lambda1_d);
  fvar<var> lambda2_v(lambda2,lambda2_d);

  fvar<var> res = log_mix(theta_v, lambda1_v, lambda2_v);
  double result = log_mix(theta_v.val_.val(), lambda1_v.val_.val(),
                          lambda2_v.val_.val());
  double deriv_denom = exp(lambda1_v.val_.val()) * theta_v.val_.val()
    + exp(lambda2_v.val_.val()) * (1 - theta_v.val_.val());
  double theta_deriv = 1 / deriv_denom * (exp(lambda1_v.val_.val()) 
        - exp(lambda2_v.val_.val()));
  double lambda1_deriv = 1 / deriv_denom * exp(lambda1_v.val_.val())
          * theta_v.val_.val();
  double lambda2_deriv = 1 / deriv_denom * exp(lambda2_v.val_.val()) 
    * (1 - theta_v.val_.val());
  double deriv = theta_deriv * theta_v.d_.val()
        + lambda2_deriv * lambda2_v.d_.val() 
        + lambda1_deriv * lambda1_v.d_.val();

  AVEC y = createAVEC(theta_v.val_, lambda1_v.val_,
     lambda2_v.val_);
  VEC g; 
  res.val_.grad(y,g);
  EXPECT_FLOAT_EQ(result, res.val_.val());
  EXPECT_FLOAT_EQ(deriv, res.d_.val());
  EXPECT_FLOAT_EQ(theta_deriv,g[0]);
  EXPECT_FLOAT_EQ(lambda1_deriv,g[1]);
  EXPECT_FLOAT_EQ(lambda2_deriv,g[2]);
}


TEST(AgradFwdLogMix, FvarVar_FvarVar_FvarVar){

  test_log_mix_3xfvar_var(0.7, 2.0, 6.0, 1.3, 2.0, 3.0);
  test_log_mix_3xfvar_var(0.7, 2.0, 6.0, 1, 0, 0);
  test_log_mix_3xfvar_var(0.7, 2.0, 6.0, 0, 1, 0);
  test_log_mix_3xfvar_var(0.7, 2.0, 6.0, 0, 0, 1);
  test_log_mix_3xfvar_var(0.7, 2.0, 6.0, 1, 0, 1);
  test_log_mix_3xfvar_var(0.7, 2.0, 6.0, 0, 1, 1);
  test_log_mix_3xfvar_var(0.7, 2.0, 6.0, 1, 1, 0);
  test_log_mix_3xfvar_var(0.7, 2.0, 6.0, 1, 1, 1);

}

/* TEST(AgradFwdLogMix, FvarVar_FvarVar_FvarVar){
  using stan::agrad::var;
  using stan::agrad::fvar;
  using std::exp;
  using ::cos;
  using ::sin;
  using stan::math::log_mix;

  fvar<var> theta_v(0.7,1.3);
  fvar<var> lambda1_v(6.0,1.0);
  fvar<var> lambda2_v(2.0,1.0);

  fvar<fvar<var> > theta_ff;
  fvar<fvar<var> > lambda1_ff;
  fvar<fvar<var> > lambda2_ff;

  theta_ff.val_.val_ = 0.7;
  theta_ff.d_.val_ = 1.0;
  theta_ff.val_.d_ = 1.0;

  lambda1_ff.val_.val_ = 6.0;
  lambda1_ff.d_.val_ = 1.0;
  lambda1_ff.val_.d_ = 1.0;

  lambda2_ff.val_.val_ = 2.0;
  lambda2_ff.d_.val_ = 1.0;
  lambda2_ff.val_.d_ = 1.0;

  fvar<var> res = log_mix(theta_v, lambda1_v, lambda2_v);
  fvar<fvar<var> > res_ff = log_mix(theta_ff, lambda1_ff, lambda2_ff);
  double result = log_mix(theta_v.val_.val(), lambda1_v.val_.val(),
                          lambda2_v.val_.val());
  EXPECT_FLOAT_EQ(result, res.val_.val());
} */
