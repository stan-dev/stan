#include <gtest/gtest.h>
#include <cmath>

#include "test/agrad/distributions/expect_eq_diffs.hpp"
#include "stan/prob/distributions/univariate/continuous/normal.hpp"
#include "stan/agrad/agrad.hpp"
#include "stan/math/special_functions.hpp"
#include "stan/meta/traits.hpp"
#include "stan/agrad/partials_vari.hpp"
#include <stan/prob/traits.hpp>


template <typename T_y, typename T_loc, typename T_scale>
void expect_propto(T_y y1, T_loc mu1, T_scale sigma1,
                   T_y y2, T_loc mu2, T_scale sigma2,
                   std::string message = "") {
  expect_eq_diffs(stan::prob::normal_log<false>(y1,mu1,sigma1),
                  stan::prob::normal_log<false>(y2,mu2,sigma2),
                  stan::prob::normal_log<true>(y1,mu1,sigma1),
                  stan::prob::normal_log<true>(y2,mu2,sigma2),
                  message);
}

using stan::agrad::var;

TEST(AgradDistributionsNormal,NoRecovery) {
  var y1 = 0.0; 
  var mu1 = -1.0;
  var sigma1 = 2.0;

  var y2 = 10.0;
  var mu2 = 7.0;
  var sigma2 = 3.0;

  expect_eq_diffs(stan::prob::normal_log<false>(y1,mu1,sigma1),
                  stan::prob::normal_log<false>(y2,mu2,sigma2),
                  stan::prob::normal_log<true>(y1,mu1,sigma1),
                  stan::prob::normal_log<true>(y2,mu2,sigma2),
                  "dummy");

  // should be recovering memory here

  var a = 1.0;
  var b = 2.0;
  var f = a * b;
  std::vector<double> g;
  std::vector<var> x;
  x.push_back(a);
  x.push_back(b);
  f.grad(x,g);
  EXPECT_FLOAT_EQ(2.0,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
}

TEST(AgradDistributionsNormal,Propto) {
  expect_propto<var,var,var>(1.0,2.0,10.0, 
                             0.1,0.0,1.0,
                             "All vars: y, mu, sigma");
}
TEST(AgradDistributionsNormal,ProptoY) {
  double mu;
  double sigma;
  
  mu = 10.0;
  sigma = 4.0;
  expect_propto<var,double,double>(20.0,mu,sigma,
                                   15.0,mu,sigma,
                                   "var: y");

}

TEST(AgradDistributionsNormal,ProptoYMu) {
  double sigma;
  sigma = 5.0;

  expect_propto<var,var,double>(20.0,15.0,sigma,
                                15.0,14.0,sigma,
                                "var: y and mu");
  
}
TEST(AgradDistributionsNormal,ProptoYSigma) {
  double mu;
  mu = -5.0;

  expect_propto<var,double,var>(-3.0,mu,4.0,
                                -6.0,mu,10.0,
                                "var: y and sigma");
}
TEST(AgradDistributionsNormal,ProptoMu) {
  double y;
  double sigma;
  
  y = 2.0;
  sigma = 10.0;
  expect_propto<double,var,double>(y,1.0,sigma,
                                   y,-1.0,sigma,
                                   "var: mu");
}
TEST(AgradDistributionsNormal,ProptoMuSigma) {
  double y;
  
  y = 2.0;
  expect_propto<double,var,var>(y,1.0,3.0,
                                   y,-1.0,4.0,
                                   "var: mu and sigma");

}
TEST(AgradDistributionsNormal,ProptoSigma) {
  double y;
  double mu;
  
  y = 2.0;
  mu = -1.0;
  expect_propto<double,double,var>(y,mu,10.0,
                                   y,mu,5.0,
                                   "var: sigma");
}
TEST(AgradDistributionsNormal,Gradient) {
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

TEST(AgradDistributionsNormal,Gradient2) {
  var y = 3.0;
  var mu = 1.0;
  var sigma = 2.0;
  
  std::vector<var> params;
  params.push_back(y);
  params.push_back(mu);
  params.push_back(sigma);

  var lp = stan::prob::normal_log<false>(y,mu,sigma);

  std::vector<double> g;
  lp.grad(params,g);
  
  EXPECT_EQ(3U,g.size());
  EXPECT_FLOAT_EQ(grad_y(3.0,1.0,2.0), g[0]);
  EXPECT_FLOAT_EQ(grad_mu(3.0,1.0,2.0), g[1]);
  EXPECT_FLOAT_EQ(grad_sigma(3.0,1.0,2.0), g[2]);
}
TEST(AgradDistributionsNormal,Gradient3) {
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

TEST(AgradDistributionsNormal,SimpleNormal) {
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

template <bool propto, typename T_y, typename T_loc, typename T_scale>
var test_normal_log(const T_y& y,
                    const T_loc& mu,
                    const T_scale& sigma) {
  using stan::prob::include_summand;
  using stan::math::pi;
  using stan::math::square;
  var lp(0.0);
  if (include_summand<propto,T_y,T_loc,T_scale>::value)
    lp -= 0.5 * (y - mu) * (y - mu) / (sigma * sigma);
  if (include_summand<propto,T_scale>::value)
    lp -= log(sigma);
  if (include_summand<propto>::value)
    lp -= log(sqrt(2.0 * pi()));
  return lp;
}

template <bool propto, typename T_y, typename T_loc, typename T_scale>
void test_gradients(double y, double mu, double sigma) {
  using stan::prob::normal_log;
  using stan::is_constant;
  T_y y1 = y;
  T_loc mu1 = mu;
  T_scale sigma1 = sigma;
  var lp1 = normal_log<true>(y1,mu1,sigma1);
  stan::agrad::grad(lp1.vi_);
  double lp1_val = lp1.val();

  double dy1 = 0.0;
  double dmu1 = 0.0;
  double dsigma1 = 0.0;
  if (!is_constant<T_y>::value)
    dy1 = var(y1).adj();
  if (!is_constant<T_loc>::value)
    dmu1 = var(mu1).adj();
  if (!is_constant<T_scale>::value)
    dsigma1 = var(sigma1).adj();

  T_y y2 = y;
  T_loc mu2 = mu;
  T_scale sigma2 = sigma;
  var lp2 = test_normal_log<true>(y2,mu2,sigma2);
  stan::agrad::grad(lp2.vi_);
  
  EXPECT_FLOAT_EQ(lp2.val(), lp1_val);
  if (!is_constant<T_y>::value)
    EXPECT_FLOAT_EQ(var(y2).adj(), dy1);
  if (!is_constant<T_loc>::value)
    EXPECT_FLOAT_EQ(var(mu2).adj(), dmu1);
  if (!is_constant<T_scale>::value)
    EXPECT_FLOAT_EQ(var(sigma2).adj(), dsigma1);
}

TEST(AgradDistributions,gradients_all) {
  using stan::agrad::var;
  test_gradients<true,var,var,var>(-2.7, -1.1, 1.3);
  test_gradients<true,var,var,double>(-2.7, -1.1, 1.3);
  test_gradients<true,var,double,var>(-2.7, -1.1, 1.3);
  test_gradients<true,var,double,double>(-2.7, -1.1, 1.3);
  test_gradients<true,double,var,var>(-2.7, -1.1, 1.3);
  test_gradients<true,double,var,double>(-2.7, -1.1, 1.3);
  test_gradients<true,double,double,var>(-2.7, -1.1, 1.3);

  test_gradients<false,var,var,var>(-2.7, -1.1, 1.3);
  test_gradients<false,var,var,double>(-2.7, -1.1, 1.3);
  test_gradients<false,var,double,var>(-2.7, -1.1, 1.3);
  test_gradients<false,var,double,double>(-2.7, -1.1, 1.3);
  test_gradients<false,double,var,var>(-2.7, -1.1, 1.3);
  test_gradients<false,double,var,double>(-2.7, -1.1, 1.3);
  test_gradients<false,double,double,var>(-2.7, -1.1, 1.3);
}

TEST(AgradDistributions,NormalCdfGrad) {
  using stan::agrad::var;
  using std::vector;
  var y = 1.0;
  var mu = 5.0;
  var sigma = 12.0;
  std::vector<var> x(3);
  x[0] = y;
  x[1] = mu;
  x[2] = sigma;

  var p = stan::prob::normal_p(y,mu,sigma);

  std::vector<double> g;
  p.grad(x,g);
  
}
