#include <cmath>
#include <gtest/gtest.h>
#include <boost/exception/all.hpp>
#include <Eigen/Dense>
#include "stan/prob/distributions.hpp"
#include "stan/maths/special_functions.hpp"


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

  var lp(0.0);
  lp += stan::prob::normal_log(x[0], mu, sigma);
  lp += stan::prob::normal_log(x[1], mu, sigma);
  lp += stan::prob::normal_log(x[2], mu, sigma);
  lp += stan::prob::normal_log(x[3], mu, sigma);
  lp += stan::prob::normal_log(x[4], mu, sigma);
  lp += stan::prob::normal_log(x[5], mu, sigma);
  lp += stan::prob::normal_log(x[6], mu, sigma);
  lp += stan::prob::normal_log(x[7], mu, sigma);
  lp += stan::prob::normal_log(x[8], mu, sigma);
  lp += stan::prob::normal_log(x[9], mu, sigma);

  EXPECT_FLOAT_EQ (-1201.418939, stan::prob::normal_log(x[0], mu, sigma).val());
  EXPECT_FLOAT_EQ (-1191.638939, stan::prob::normal_log(x[1], mu, sigma).val());
  EXPECT_FLOAT_EQ (-1196.523939, stan::prob::normal_log(x[2], mu, sigma).val());
  EXPECT_FLOAT_EQ (-1206.323939, stan::prob::normal_log(x[3], mu, sigma).val());
  EXPECT_FLOAT_EQ (-1196.523939, stan::prob::normal_log(x[4], mu, sigma).val());
  EXPECT_FLOAT_EQ (-1194.080189, stan::prob::normal_log(x[5], mu, sigma).val());
  EXPECT_FLOAT_EQ (-1186.276989, stan::prob::normal_log(x[6], mu, sigma).val());
  EXPECT_FLOAT_EQ (-1203.870189, stan::prob::normal_log(x[7], mu, sigma).val());
  EXPECT_FLOAT_EQ (-1195.546139, stan::prob::normal_log(x[8], mu, sigma).val());
  EXPECT_FLOAT_EQ (-1200.439139, stan::prob::normal_log(x[9], mu, sigma).val());
  EXPECT_FLOAT_EQ (-11972.64, lp.val());

  std::vector<double> g;
  std::vector<var> params;
  params.resize(2);
  params[0] = mu;
  params[1] = sigma;
  lp.grad(params, g);
  EXPECT_EQ (2U, g.size());
  EXPECT_FLOAT_EQ ((49.0+48.8+48.9+49.1+48.9+48.85+48.69+49.05+48.88+48.98), g[0]);
  EXPECT_FLOAT_EQ ((49.0*49.0+48.8*48.8+48.9*48.9+49.1*49.1+48.9*48.9+48.85*48.85+48.69*48.69+49.05*49.05+48.88*48.88+48.98*48.98) - 10.0, g[1]);
}
