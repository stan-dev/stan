#include <gtest/gtest.h>
#include "stan/prob/distributions_normal.hpp"
#include "stan/agrad/agrad.hpp"

using stan::agrad::var;
TEST(AgradDistributions,Normal) {
  var y;
  var mu;
  var sigma;
  
  y = 0.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-0.9189385, stan::prob::normal_log(y,mu,sigma).val());
  
  y = 1.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-1.418939,  stan::prob::normal_log(y,mu,sigma).val());
  
  y = -2.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-2.918939,  stan::prob::normal_log(y,mu,sigma).val());
 
  y = -3.5;
  mu = 1.9;
  sigma = 7.2;
  EXPECT_FLOAT_EQ(-3.174270,  stan::prob::normal_log(y,mu,sigma).val());
}
TEST(AgradDistributionsPropto,Normal) {
  double LOG_SQRT_TWO_PI = 0.9189385;
  var y;
  var mu;
  var sigma;
  
  y = 0.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-0.9189385+LOG_SQRT_TWO_PI, stan::prob::normal_log<true>(y,mu,sigma).val());
  
  y = 1.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-1.41893853+LOG_SQRT_TWO_PI,  stan::prob::normal_log<true>(y,mu,sigma).val());
  
  y = -2.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-2.918939+LOG_SQRT_TWO_PI,  stan::prob::normal_log<true>(y,mu,sigma).val());
 
  y = -3.5;
  mu = 1.9;
  sigma = 7.2;
  EXPECT_FLOAT_EQ(-3.174270+LOG_SQRT_TWO_PI,  stan::prob::normal_log<true>(y,mu,sigma).val());
}
TEST(AgradDistributionsPropto,NormalY) {
  double LOG_SQRT_TWO_PI = 0.9189385;
  double y;
  var mu;
  var sigma;
  
  y = 0.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-0.9189385+LOG_SQRT_TWO_PI, stan::prob::normal_log<true>(y,mu,sigma).val());
  
  y = 1.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-1.41893853+LOG_SQRT_TWO_PI,  stan::prob::normal_log<true>(y,mu,sigma).val());
  
  y = -2.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-2.918939+LOG_SQRT_TWO_PI,  stan::prob::normal_log<true>(y,mu,sigma).val());
 
  y = -3.5;
  mu = 1.9;
  sigma = 7.2;
  EXPECT_FLOAT_EQ(-3.174270+LOG_SQRT_TWO_PI,  stan::prob::normal_log<true>(y,mu,sigma).val());
}
TEST(AgradDistributionsPropto,NormalMu) {
  double LOG_SQRT_TWO_PI = 0.9189385;
  var y;
  double mu;
  var sigma;
  
  y = 0.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-0.9189385+LOG_SQRT_TWO_PI, stan::prob::normal_log<true>(y,mu,sigma).val());
  
  y = 1.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-1.41893853+LOG_SQRT_TWO_PI,  stan::prob::normal_log<true>(y,mu,sigma).val());
  
  y = -2.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-2.918939+LOG_SQRT_TWO_PI,  stan::prob::normal_log<true>(y,mu,sigma).val());
 
  y = -3.5;
  mu = 1.9;
  sigma = 7.2;
  EXPECT_FLOAT_EQ(-3.174270+LOG_SQRT_TWO_PI,  stan::prob::normal_log<true>(y,mu,sigma).val());
}
TEST(AgradDistributionsPropto,NormalSigma) {
  double LOG_SQRT_TWO_PI = 0.9189385;
  var y;
  var mu;
  double sigma;
  
  y = 0.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-0.9189385+LOG_SQRT_TWO_PI+log(sigma), stan::prob::normal_log<true>(y,mu,sigma).val());
  
  y = 1.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-1.41893853+LOG_SQRT_TWO_PI+log(sigma),  stan::prob::normal_log<true>(y,mu,sigma).val());
  
  y = -2.0;
  mu = 0.0;
  sigma = 1.0;
  EXPECT_FLOAT_EQ(-2.918939+LOG_SQRT_TWO_PI+log(sigma),  stan::prob::normal_log<true>(y,mu,sigma).val());
 
  y = -3.5;
  mu = 1.9;
  sigma = 7.2;
  EXPECT_NEAR(-3.1742756+LOG_SQRT_TWO_PI+log(sigma),  stan::prob::normal_log<true>(y,mu,sigma).val(),0.00001);
}
