#include <cmath>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "stan/maths/special_functions.hpp"
#include "stan/prob/distributions.hpp"
#include "stan/agrad/matrix.hpp"

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;

using stan::agrad::var;

TEST(prob_prob,uniform) {
  EXPECT_FLOAT_EQ(1.0, exp(stan::prob::uniform_log(0.2,0.0,1.0)));
  EXPECT_FLOAT_EQ(2.0, exp(stan::prob::uniform_log(0.2,-0.25,0.25)));
  EXPECT_FLOAT_EQ(0.1, exp(stan::prob::uniform_log(101.0,100.0,110.0)));
  // lower boundary
  EXPECT_FLOAT_EQ(1.0, exp(stan::prob::uniform_log(0.0,0.0,1.0)));
  // upper boundary
  EXPECT_FLOAT_EQ(1.0, exp(stan::prob::uniform_log(1.0,0.0,1.0)));
}

TEST(prob_prob,norm_p) {
  // values from R pnorm()
  EXPECT_FLOAT_EQ(0.5000000, stan::prob::normal_p (0.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.8413447, stan::prob::normal_p (1.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(0.4012937, stan::prob::normal_p (1.0, 2.0, 4.0));
}
TEST(prob_prob,norm) {
  // values from R dnorm()
  EXPECT_FLOAT_EQ(-0.9189385, stan::prob::normal_log(0.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-1.418939,  stan::prob::normal_log(1.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-2.918939,  stan::prob::normal_log(-2.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-3.174270,  stan::prob::normal_log(-3.5,1.9,7.2));
}
TEST(prob_prob,norm_exception) {
  double sigma_d = 0.0;
  var sigma_v = 0.0;
  EXPECT_THROW(stan::prob::normal_log(0.0,0.0,sigma_d), std::domain_error);
  EXPECT_THROW(stan::prob::normal_log(0.0,0.0,sigma_v), std::domain_error);
  sigma_d = -1.0;
  sigma_v = -1.0;
  EXPECT_THROW(stan::prob::normal_log(0.0,0.0,sigma_d), std::domain_error);
  EXPECT_THROW(stan::prob::normal_log(0.0,0.0,sigma_v), std::domain_error);
}
TEST(prob_prob,norm_trunc_lh) {
  // values from R dnorm()
  double mu;
  double sigma;
  double low;
  double high;
  
  mu = 0;
  sigma = 1.0;
  low = -2.0;
  high = 1.0;
  // mu <- 0; sigma <- 1.0; low <- -2.0; high <- 1.0;
  // R: log ( 0.0 )
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), stan::prob::normal_trunc_lh_log(-5.0, mu, sigma, low, high));
  // R: log ( dnorm(-2.0, mu, sigma) / (pnorm (high, mu, sigma) - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-2.718772, stan::prob::normal_trunc_lh_log(-2.0, mu, sigma, low, high));
  // R: log ( dnorm(1.0, mu, sigma) / (pnorm (high, mu, sigma) - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-1.218772, stan::prob::normal_trunc_lh_log( 1.0, mu, sigma, low, high));
  // R: log ( 0.0 )
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), stan::prob::normal_trunc_lh_log(10.0, mu, sigma, low, high));

  // R: log ( dnorm(0.0, mu, sigma) / (pnorm (high, mu, sigma) - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-0.7187722, stan::prob::normal_trunc_lh_log(0.0, mu, sigma, low, high));
  // R: log ( dnorm(0.5, mu, sigma) / (pnorm (high, mu, sigma) - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-0.8437722, stan::prob::normal_trunc_lh_log(0.5, mu, sigma, low, high));
  // R: log ( dnorm(-0.5, mu, sigma) / (pnorm (high, mu, sigma) - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-0.8437722, stan::prob::normal_trunc_lh_log(-0.5, mu, sigma, low, high));
}
TEST(prob_prob,norm_trunc_lh_exception) {
  double y = 0;
  double mu = 0;
  double sigma = 1;
  double low = -5;
  double high = 5;
  EXPECT_THROW(stan::prob::normal_trunc_lh_log(y, mu, 0.0, low, high), std::domain_error);
  EXPECT_THROW(stan::prob::normal_trunc_lh_log(y, mu, -1.0, low, high), std::domain_error);
  EXPECT_THROW(stan::prob::normal_trunc_lh_log(y, mu, sigma, high, low), std::invalid_argument);
  EXPECT_NO_THROW(stan::prob::normal_trunc_lh_log(y, mu, sigma, low, low));
}
TEST(prob_prob,norm_trunc_l) {
  // values from R dnorm()
  double mu;
  double sigma;
  double low;
  
  mu = 0;
  sigma = 1.0;
  low = -2.0;
  // mu <- 0; sigma <- 1.0; low <- -2.0; 
  // R: log ( 0.0 )
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), stan::prob::normal_trunc_l_log(-5.0, mu, sigma, low));
  // R: log ( dnorm(-2.0, mu, sigma) / (1 - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-2.895926, stan::prob::normal_trunc_l_log(-2.0, mu, sigma, low));
  // R: log ( dnorm(1.0, mu, sigma) / (1 - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-1.395926, stan::prob::normal_trunc_l_log( 1.0, mu, sigma, low));
  // R: log ( dnorm(10.0, mu, sigma) / (1 - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-50.89593, stan::prob::normal_trunc_l_log(10.0, mu, sigma, low));

  // R: log ( dnorm(0.0, mu, sigma) / (1 - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-0.8959256, stan::prob::normal_trunc_l_log(0.0, mu, sigma, low));
  // R: log ( dnorm(0.5, mu, sigma) / (1 - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-1.020926, stan::prob::normal_trunc_l_log(0.5, mu, sigma, low));
  // R: log ( dnorm(-0.5, mu, sigma) / (1 - pnorm(low, mu, sigma)) )
  EXPECT_FLOAT_EQ(-1.020926, stan::prob::normal_trunc_l_log(-0.5, mu, sigma, low));
}
TEST(prob_prob,norm_trunc_l_exception) {
  double y = 0;
  double mu = 0;
  double sigma = 1;
  double low = -5;
  EXPECT_NO_THROW(stan::prob::normal_trunc_l_log(y, mu, sigma, low));
  EXPECT_THROW(stan::prob::normal_trunc_l_log(y, mu, 0.0, low), std::domain_error);
  EXPECT_THROW(stan::prob::normal_trunc_l_log(y, mu, -1.0, low), std::domain_error);
}
TEST(prob_prob,norm_trunc_h) {
  // values from R dnorm()
  double mu;
  double sigma;
  double high;
  
  mu = 0;
  sigma = 1.0;
  high = 1.0;
  // mu <- 0; sigma <- 1.0; high <- 1.0
  // R: log ( 0.0 )
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), stan::prob::normal_trunc_h_log(5.0, mu, sigma, high));
  // R: log ( dnorm(-2.0, mu, sigma) / pnorm(high, mu, sigma) )
  EXPECT_FLOAT_EQ(-2.746185, stan::prob::normal_trunc_h_log(-2.0, mu, sigma, high));
  // R: log ( dnorm(1.0, mu, sigma) / pnorm(high, mu, sigma) )
  EXPECT_FLOAT_EQ(-1.246185, stan::prob::normal_trunc_h_log( 1.0, mu, sigma, high));

  // R: log ( dnorm(0.0, mu, sigma) / pnorm(high, mu, sigma) )
  EXPECT_FLOAT_EQ(-0.7461848, stan::prob::normal_trunc_h_log(0.0, mu, sigma, high));
  // R: log ( dnorm(0.5, mu, sigma) / pnorm(high, mu, sigma) )
  EXPECT_FLOAT_EQ(-0.8711848, stan::prob::normal_trunc_h_log(0.5, mu, sigma, high));
  // R: log ( dnorm(-0.5, mu, sigma) / pnorm(high, mu, sigma) )
  EXPECT_FLOAT_EQ(-0.8711848, stan::prob::normal_trunc_h_log(-0.5, mu, sigma, high));
}
TEST(prob_prob,norm_trunc_h_exception) {
  double y = 0;
  double mu = 0;
  double sigma = 1;
  double high = -5;
  EXPECT_NO_THROW(stan::prob::normal_trunc_h_log(y, mu, sigma, high));
  EXPECT_THROW(stan::prob::normal_trunc_h_log(y, mu, 0.0, high), std::domain_error);
  EXPECT_THROW(stan::prob::normal_trunc_h_log(y, mu, -1.0, high), std::domain_error);
}


TEST(prob_prob,gamma) {
  EXPECT_FLOAT_EQ(-0.6137056, stan::prob::gamma_log(1.0,2.0,2.0));
  EXPECT_FLOAT_EQ(-3.379803, stan::prob::gamma_log(2.0,0.25,0.75));
  EXPECT_FLOAT_EQ(-1, stan::prob::gamma_log(1,1,1));
}


TEST(prob_prob,inv_gamma) {
  EXPECT_FLOAT_EQ(-1, stan::prob::inv_gamma_log(1,1,1.0));
  EXPECT_FLOAT_EQ(-0.8185295, stan::prob::inv_gamma_log(0.5,2.9,3.1));
}

TEST(prob_prob,chi_square) {
  EXPECT_FLOAT_EQ(-3.835507, stan::prob::chi_square_log(7.9,3.0));
  EXPECT_FLOAT_EQ(-2.8927, stan::prob::chi_square_log(1.9,0.5));
}

TEST(prob_prob,inv_chi_square) {
  EXPECT_FLOAT_EQ(-0.3068528, stan::prob::inv_chi_square_log(0.5,2.0));
  EXPECT_FLOAT_EQ(-12.28905, stan::prob::inv_chi_square_log(3.2,9.1));
}

TEST(prob_prob,scaled_inv_chi_square) {
  EXPECT_FLOAT_EQ(-3.091965, stan::prob::scaled_inv_chi_square_log(12.7,6.1,3.0));
  EXPECT_FLOAT_EQ(-1.737086, stan::prob::scaled_inv_chi_square_log(1.0,1.0,0.5));
}

TEST(prob_prob,exponential) {
  EXPECT_FLOAT_EQ(-2.594535, stan::prob::exponential_log(2.0,1.5));
  EXPECT_FLOAT_EQ(-57.13902, stan::prob::exponential_log(15.0,3.9));
}

TEST(prob_prob,cauchy) {
  EXPECT_FLOAT_EQ(-1.837877, stan::prob::cauchy_log(1.0, 0.0, 1.0));
  EXPECT_FLOAT_EQ(-2.323385, stan::prob::cauchy_log(-1.5, 0.0, 1.0));
  EXPECT_FLOAT_EQ(-2.323385, stan::prob::cauchy_log(-2.5, -1.0, 1.0));
  // need test with scale != 1
}

TEST(prob_prob,student_t) {
  EXPECT_FLOAT_EQ(-1.837877, stan::prob::student_t_log(1.0,1.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-3.596843, stan::prob::student_t_log(-3.0,2.0,0.0,1.0));
  EXPECT_FLOAT_EQ(-2.531024, stan::prob::student_t_log(2.0,1.0,0.0,2.0));
  // need test with scale != 1
}

TEST(prob_prob,beta) {
  EXPECT_FLOAT_EQ(0.0, stan::prob::beta_log(0.2,1.0,1.0));
  EXPECT_FLOAT_EQ(1.628758, stan::prob::beta_log(0.3,12.0,25.0));
}

TEST(prob_prob,pareto) {
  EXPECT_FLOAT_EQ(-1.909543, stan::prob::pareto_log(1.5,0.5,2.0));
  EXPECT_FLOAT_EQ(-25.69865, stan::prob::pareto_log(19.5,0.15,5.0));
}

TEST(prob_prob,double_exponential) {
  EXPECT_FLOAT_EQ(-0.6931472, stan::prob::double_exponential_log(1.0,1.0,1.0));
  EXPECT_FLOAT_EQ(-1.693147, stan::prob::double_exponential_log(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(-5.693147, stan::prob::double_exponential_log(-3.0,2.0,1.0));
  EXPECT_FLOAT_EQ(-1.886294, stan::prob::double_exponential_log(1.0,0.0,2.0));
  EXPECT_FLOAT_EQ(-0.8, stan::prob::double_exponential_log(1.9,2.3,0.5));
  EXPECT_FLOAT_EQ(-0.9068528, stan::prob::double_exponential_log(1.9,2.3,0.25));
}

TEST(prob_prob,weibull) {
  EXPECT_FLOAT_EQ(-2.0, stan::prob::weibull_log(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(-3.277094, stan::prob::weibull_log(0.25,2.9,1.8));
  EXPECT_FLOAT_EQ(-102.8962, stan::prob::weibull_log(3.9,1.7,0.25));
}

TEST(prob_prob,logistic) {
  EXPECT_FLOAT_EQ(-2.129645, stan::prob::logistic_log(1.2,0.3,2.0));
  EXPECT_FLOAT_EQ(-3.430098, stan::prob::logistic_log(-1.0,0.2,0.25));
}

TEST(prob_prob,lognormal) {
  EXPECT_FLOAT_EQ(-1.509803, stan::prob::lognormal_log(1.2,0.3,1.5));
  EXPECT_FLOAT_EQ(-3.462263, stan::prob::lognormal_log(12.0,3.0,0.9));
}

TEST(prob_prob,dirichlet) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  Matrix<double,Dynamic,1> alpha(3,1);
  alpha << 1.0, 1.0, 1.0;
  EXPECT_FLOAT_EQ(0.6931472, stan::prob::dirichlet_log(theta,alpha));
  
  Matrix<double,Dynamic,1> theta2(4,1);
  theta2 << 0.01, 0.01, 0.8, 0.18;
  Matrix<double,Dynamic,1> alpha2(4,1);
  alpha2 << 10.5, 11.5, 19.3, 5.1;
  EXPECT_FLOAT_EQ(-43.40045, stan::prob::dirichlet_log(theta2,alpha2));
}

TEST(prob_prob,multi_normal) {
  Matrix<double,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<double,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  EXPECT_FLOAT_EQ(-11.73908, stan::prob::multi_normal_log<double>(y,mu,Sigma));
}

TEST(prob_prob,bernoulli) {
  EXPECT_FLOAT_EQ(std::log(0.25), stan::prob::bernoulli_log(1,0.25));
  EXPECT_FLOAT_EQ(std::log(1.0 - 0.25), stan::prob::bernoulli_log(0,0.25));
}

TEST(prob_prob,binomial) {
  EXPECT_FLOAT_EQ(-2.144372, stan::prob::binomial_log(10,20,0.4));
  EXPECT_FLOAT_EQ(-16.09438, stan::prob::binomial_log(0,10,0.8));
}

TEST(prob_prob,poisson) {
  EXPECT_FLOAT_EQ(-2.900934, stan::prob::poisson_log(17,13.0));
  EXPECT_FLOAT_EQ(-145.3547, stan::prob::poisson_log(192,42.0));
}

TEST(prob_prob,neg_binomial) {
  EXPECT_FLOAT_EQ(-7.786663, stan::prob::neg_binomial_log(10,2.0,1.5));
  EXPECT_FLOAT_EQ(-142.6147, stan::prob::neg_binomial_log(100,3.0,3.5));
}

TEST(prob_prob,beta_binomial) {
  EXPECT_FLOAT_EQ(-1.854007, stan::prob::beta_binomial_log(5,20,10.0,25.0));
  EXPECT_FLOAT_EQ(-4.376696, stan::prob::beta_binomial_log(25,100,30.0,50.0));
}

TEST(prob_prob,categorical) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.3, 0.5, 0.2;
  EXPECT_FLOAT_EQ(-1.203973, stan::prob::categorical_log(0,theta));
  EXPECT_FLOAT_EQ(-0.6931472, stan::prob::categorical_log(1,theta));
}

TEST(prob_prob,hypergeometric) {
  EXPECT_FLOAT_EQ(-4.119424, stan::prob::hypergeometric_log(5,15,10,10));
  EXPECT_FLOAT_EQ(-2.302585, stan::prob::hypergeometric_log(0,2,3,2));
}

TEST(prob_prob,multinomial) {
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  EXPECT_FLOAT_EQ(-2.002481, stan::prob::multinomial_log(ns,theta));
}

TEST(prob_prob,wishart_1) {
  Matrix<double,Dynamic,Dynamic> Sigma(2,2);
  Sigma << 1.848220, 1.899623, 
    1.899623, 12.751941;

  Matrix<double,Dynamic,Dynamic> Y(2,2);
  Y <<  2.011108, -11.20661,
    -11.206611, 112.94139;

  unsigned int dof = 3;
  
  double lp = log(8.658e-07); // computed with MCMCpack in R
 
  EXPECT_NEAR(lp, stan::prob::wishart_log(Y,dof,Sigma), 0.01);
}

TEST(prob_prob,wishart) {
  Matrix<double,Dynamic,Dynamic> Y(4,4);
  Y << 7.988168,  -9.555605, -14.47483,   4.395895,
    -9.555605,  44.750570,  49.21577, -18.454186,
    -14.474830,  49.215769,  60.08987, -21.481079,
    4.395895, -18.454186, -21.48108, 7.885833;
  
  Matrix<double,Dynamic,Dynamic> Sigma(4,4);
  Sigma << 2.9983662,  0.2898776, -2.650523,  0.1055911,
    0.2898776, 11.4803610,  7.157993, -3.1129955,
    -2.6505229,  7.1579931, 11.676181, -3.5866852,
    0.1055911, -3.1129955, -3.586685,  1.4482736;

  double dof = 4;
  double log_p = log(8.034197e-10);
  EXPECT_NEAR(log_p, stan::prob::wishart_log(Y,dof,Sigma),0.01);
  
  dof = 5;
  log_p = log(1.517951e-10);
  EXPECT_NEAR(log_p, stan::prob::wishart_log(Y,dof,Sigma),0.01);
}

TEST(prob_prob,iwishart) {
  Matrix<double,Dynamic,Dynamic> Y(3,3);
  Y <<  12.147233, -11.9036079, 1.0910458,
    -11.903608,  16.7585782, 0.8530256,
     1.091046,   0.8530256, 2.5786609;

  Matrix<double,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 7.785215,  3.0597878,  1.1071663,
    3.059788, 10.3515035, -0.1232598,
    1.107166, -0.1232598,  7.7623386;
  
  double dof = 4.0;
  double log_p = log(2.008407e-08);

  EXPECT_NEAR(log_p, stan::prob::inv_wishart_log(Y,dof,Sigma), 0.01);
}
