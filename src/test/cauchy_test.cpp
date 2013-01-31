#include <stan/prob/distributions/univariate/continuous/cauchy.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/agrad.hpp>

TEST(Cauchy, DISABLED_Test_ddd) {
  using stan::agrad::var;
  using stan::prob::cauchy_cdf;
  
  double y, mu, sigma;
  y = 1.0;
  mu = 0.0;
  sigma = 1.0;
  
  double cdf = cauchy_cdf(y, mu, sigma);
  EXPECT_FLOAT_EQ(0.75, cdf);
}

TEST(Cauchy, Test_ddv) {
  using stan::agrad::var;
  using stan::prob::cauchy_cdf;
  using std::vector;

  double y, mu;
  y = 1.0;
  mu = 0.0;
  var sigma = 1.0;
  stan::agrad::print_stack(std::cout);
  
  //var cdf = cauchy_cdf(y, mu, sigma);
  var cdf = stan::agrad::atan2(1.0, sigma) / 3.14;
  //var cdf = stan::agrad::atan2(y-mu, sigma) /  3.14;
  //EXPECT_FLOAT_EQ(0.75, cdf.val());

  stan::agrad::print_stack(std::cout);


  vector<var> x;
  x.push_back(sigma);
  vector<double> gradients;

  std::cout << "x[0]: " << x[0] << std::endl;
  std::cout << "x[0]: " << x[0].val() << std::endl;

  cdf.grad(x, gradients);
 
 
  std::cout << "x[0]: " << x[0] << std::endl;  
  std::cout << "x[0]: " << x[0].val() << std::endl;
  std::cout << "gradient[0]: " << gradients[0] << std::endl;
}
