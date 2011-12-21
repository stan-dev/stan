#include <cmath>
#include <gtest/gtest.h>
#include <boost/exception/all.hpp>
#include <Eigen/Dense>
#include "stan/agrad/agrad.hpp"
#include "stan/maths/special_functions.hpp"
#include "stan/prob/distributions.hpp"

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;

using stan::agrad::var;

TEST(prob_prob,norm_exception) {
  var sigma_v = 0.0;
  EXPECT_THROW(stan::prob::normal_log(0.0,0.0,sigma_v), std::domain_error);
  sigma_v = -1.0;
  EXPECT_THROW(stan::prob::normal_log(0.0,0.0,sigma_v), std::domain_error);
}
TEST(agrad_agrad,norm_grad) {
  var mu = 3.0;
  var sigma = 2.0;
  std::vector<double> x;
  x.resize(2);
  x[0] = 0.0;
  x[1] = 1.0;

  var lp(0.0);
  lp += 0.0;
  lp += stan::prob::normal_log(x[0], mu, sigma);
  lp += stan::prob::normal_log(x[1], mu, sigma);

  EXPECT_FLOAT_EQ (-2.7370857, stan::prob::normal_log (x[0], mu, sigma).val());
  EXPECT_FLOAT_EQ (-2.1120857, stan::prob::normal_log (x[1], mu, sigma).val());
  EXPECT_FLOAT_EQ (-4.8491714, lp.val());

  std::vector<double> g;
  std::vector<var> params;
  params.resize(2);
  params[0] = mu;
  params[1] = sigma;
  lp.grad(params, g);
  
  EXPECT_EQ (2U, g.size());
  EXPECT_FLOAT_EQ (((0.0 - 3.0) + (1.0 - 3.0)) / (2.0 * 2.0), g[0]);
  EXPECT_FLOAT_EQ (-2/2.0 + (9.0 + 4.0)/8.0, g[1]);
}

double grad_y(double y, double mu, double sigma) {
  return (mu - y) / (sigma * sigma);
}
double grad_mu(double y, double mu, double sigma) {
  return (y - mu) / (sigma * sigma);
}
double grad_sigma(double y, double mu, double sigma) {
  return - 1/sigma + (y * y - 2 * y * mu + mu * mu) / (sigma * sigma * sigma);
}

TEST(agrad_agrad,norm_example_2) {
  var y = 3.0;
  var mu = 1.0;
  var sigma = 2.0;
  var lp = stan::prob::normal_log(y,mu,sigma);
  std::vector<var> params;
  params.push_back(y);
  params.push_back(mu);
  params.push_back(sigma);
  std::vector<double> g;
  lp.grad(params,g);
  
  EXPECT_EQ(3U,g.size());
  EXPECT_FLOAT_EQ(grad_y(3.0,1.0,2.0), g[0]);
  EXPECT_FLOAT_EQ(grad_mu(3.0,1.0,2.0), g[1]);
  EXPECT_FLOAT_EQ(grad_sigma(3.0,1.0,2.0), g[2]);
}


TEST(agrad_agrad,norm_example_3) {
  var y = -2.7;
  var mu = 1.9;
  var sigma = 3.3;
  var lp = stan::prob::normal_log(y,mu,sigma);
  std::vector<var> params;
  params.push_back(y);
  params.push_back(mu);
  params.push_back(sigma);
  std::vector<double> g;
  lp.grad(params,g);
  
  EXPECT_EQ(3U,g.size());
  EXPECT_FLOAT_EQ(grad_y(y.val(),mu.val(),sigma.val()), g[0]);
  EXPECT_FLOAT_EQ(grad_mu(y.val(),mu.val(),sigma.val()), g[1]);
  EXPECT_FLOAT_EQ(grad_sigma(y.val(),mu.val(),sigma.val()), g[2]);
}

TEST(agrad_agrad,really_simple) {
  var x = 0.0;
  var y = 1.0;
  x += y;
  std::vector<var> params;
  params.push_back(y);
  std::vector<double> g;
  x.grad(params,g);
  EXPECT_FLOAT_EQ(1.0,g[0]);
}

TEST(agrad_agrad,norm_grad_small_example) {
  var mu = 0;
  var sigma = 1;
  std::vector<double> x;
  x.resize(10);
  x[0] = 49;
  x[1] = 48.8; 
  x[2] = 48.9; 
  x[3] = 49.1;
  x[4] = 48.9;
  x[5] = 48.85;
  x[6] = 48.69;
  x[7] = 49.05;
  x[8] = 48.88; 
  x[9] = 48.98;

  unsigned int N = 10;

  var lp(0.0);
  for (unsigned int n = 0; n < N; ++n)
    lp += stan::prob::normal_log(x[n], mu, sigma);

  EXPECT_FLOAT_EQ (-11972.64, lp.val());

  std::vector<double> g;
  std::vector<var> params;
  params.push_back(mu);
  params.push_back(sigma);
  lp.grad(params, g);
  EXPECT_EQ (2U, g.size());
  
  double dmu = 0.0;
  for (unsigned int n = 0; n < N; ++n)
    dmu += grad_mu(x[n], 0.0, 1.0);
  EXPECT_FLOAT_EQ(dmu,g[0]);

  double dsigma = 0.0;
  for (unsigned int n = 0; n < N; ++n)
    dsigma += grad_sigma(x[n], 0.0, 1.0);
  EXPECT_FLOAT_EQ(dsigma,g[1]);

}
