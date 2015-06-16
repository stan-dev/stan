#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/log_mix.hpp>
#include <stan/math/prim/mat/err/constraint_tolerance.hpp>
#include <test/unit/math/fwd/scal/fun/nan_util.hpp>
#include <cmath>
#include <typeinfo>
#include <stan/math/fwd/scal/fun/log_mix.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/sin.hpp>

void test_log_mix_fff(double theta, double lambda1, double lambda2,
                      double theta_d, double lambda1_d, double lambda2_d){
  using stan::math::fvar;
  using stan::math::log_mix;
  using ::exp;
  using ::log;

  fvar<double> theta_f(theta,theta_d);
  fvar<double> lambda1_f(lambda1,lambda1_d);
  fvar<double> lambda2_f(lambda2,lambda2_d);

  fvar<double> f = log_mix(theta_f, lambda1_f, lambda2_f);
  fvar<double> f2 = log(theta_f * exp(lambda1_f) + (1 - theta_f) * exp(lambda2_f));
  EXPECT_FLOAT_EQ(f.val_,f2.val_);
  EXPECT_FLOAT_EQ(f.d_,f2.d_);

  fvar<double> theta_f_invalid(-1.0,theta_d);
  EXPECT_THROW(log_mix(theta_f_invalid,lambda1_f,lambda2_f),std::domain_error);
}

void test_log_mix_f_explicit(double theta, double lambda1, double x){
  using stan::math::fvar;
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
  using stan::math::fvar;
  using stan::math::log_mix;
  using ::exp;

  fvar<double> theta_f(theta,theta_d);
  fvar<double> lambda1_f(lambda1,lambda1_d);

  fvar<double> f = log_mix(theta_f, lambda1_f, lambda2);
  fvar<double> f2 = log(theta_f * exp(lambda1_f) + (1 - theta_f) * exp(lambda2));

  EXPECT_FLOAT_EQ(f.val_,f2.val_);
  EXPECT_FLOAT_EQ(f.d_,f2.d_);

  fvar<double> theta_f_invalid(-1.0,theta_d);
  EXPECT_THROW(log_mix(theta_f_invalid,lambda1_f,lambda2),std::domain_error);
}

void test_log_mix_ff_ex_lam_1(double theta, double lambda1, double lambda2,
                              double theta_d, double lambda2_d){
  using stan::math::fvar;
  using stan::math::log_mix;

  fvar<double> theta_f(theta,theta_d);
  fvar<double> lambda2_f(lambda2,lambda2_d);

  fvar<double> f = log_mix(theta_f, lambda1, lambda2_f);
  fvar<double> f2 = log(theta_f * exp(lambda1) + (1 - theta_f) * exp(lambda2_f));
  EXPECT_FLOAT_EQ(f.val_,f2.val_);
  EXPECT_FLOAT_EQ(f.d_,f2.d_);

  fvar<double> theta_f_invalid(-1.0,theta_d);
  EXPECT_THROW(log_mix(theta_f_invalid,lambda1,lambda2_f),std::domain_error);
}

void test_log_mix_ff_ex_theta(double theta, double lambda1, double lambda2,
                              double lambda1_d, double lambda2_d){
  using stan::math::fvar;
  using stan::math::log_mix;

  fvar<double> lambda1_f(lambda1,lambda1_d);
  fvar<double> lambda2_f(lambda2,lambda2_d);

  fvar<double> f = log_mix(theta, lambda1_f, lambda2_f);
  fvar<double> f2 = log(theta * exp(lambda1_f) + (1 - theta) * exp(lambda2_f));
  EXPECT_FLOAT_EQ(f.val_,f2.val_);
  EXPECT_FLOAT_EQ(f.d_,f2.d_);

  EXPECT_THROW(log_mix(-1.0,lambda1_f,lambda2_f),std::domain_error);
}

void test_log_mix_f_theta(double theta, double lambda1, double lambda2,
                          double theta_d){
  using stan::math::fvar;
  using stan::math::log_mix;

  fvar<double> theta_f(theta,theta_d);

  fvar<double> f = log_mix(theta_f, lambda1, lambda2);
  fvar<double> f2 = log(theta_f * exp(lambda1) + (1 - theta_f) * exp(lambda2));
  EXPECT_FLOAT_EQ(f.val_,f2.val_);
  EXPECT_FLOAT_EQ(f.d_,f2.d_);

  fvar<double> theta_f_invalid(-1.0,theta_d);
  EXPECT_THROW(log_mix(theta_f_invalid,lambda1,lambda2),std::domain_error);
}

void test_log_mix_f_lam_1(double theta, double lambda1, double lambda2,
                          double lambda1_d){
  using stan::math::fvar;
  using stan::math::log_mix;

  fvar<double> lambda1_f(lambda1,lambda1_d);

  fvar<double> f = log_mix(theta, lambda1_f, lambda2);
  fvar<double> f2 = log(theta * exp(lambda1_f) + (1 - theta) * exp(lambda2));
  EXPECT_FLOAT_EQ(f.val_,f2.val_);
  EXPECT_FLOAT_EQ(f.d_,f2.d_);

  EXPECT_THROW(log_mix(-1.0,lambda1_f,lambda2),std::domain_error);
}

void test_log_mix_f_lam_2(double theta, double lambda1, double lambda2,
                          double lambda2_d){
  using stan::math::fvar;
  using stan::math::log_mix;

  fvar<double> lambda2_f(lambda2,lambda2_d);

  fvar<double> f = log_mix(theta, lambda1, lambda2_f);
  fvar<double> f2 = log(theta * exp(lambda1) + (1 - theta) * exp(lambda2_f));
  EXPECT_FLOAT_EQ(f.val_,f2.val_);
  EXPECT_FLOAT_EQ(f.d_,f2.d_);

  EXPECT_THROW(log_mix(-1.0,lambda1,lambda2_f),std::domain_error);
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

  test_log_mix_f_theta(0.7, 1.5, -2.0, 1);
  test_log_mix_f_theta(0.7, 1.5, -2.0, 0);

  test_log_mix_f_theta(0.7, -1.5, 2.0, 1);
  test_log_mix_f_theta(0.7, -1.5, 2.0, 0);

  test_log_mix_f_lam_1(0.7, 1.5, -2.0, 1);
  test_log_mix_f_lam_1(0.7, 1.5, -2.0, 0);

  test_log_mix_f_lam_1(0.7, -1.5, 2.0, 1);
  test_log_mix_f_lam_1(0.7, -1.5, 2.0, 0);

  test_log_mix_f_lam_2(0.7, 1.5, -2.0, 1);
  test_log_mix_f_lam_2(0.7, 1.5, -2.0, 0);

  test_log_mix_f_lam_2(0.7, -1.5, 2.0, 1);
  test_log_mix_f_lam_2(0.7, -1.5, 2.0, 0);

  test_log_mix_f_explicit(0.7, 1.5, 5);
  test_log_mix_f_explicit(0.7, 0.1, 5);
  test_log_mix_f_explicit(0.999, 0.1, 5);
  test_log_mix_f_explicit(0.0001, 0.1, 5); 
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
  test_nan_fwd(log_mix_,0.7,3.0,5.0,true);
}
