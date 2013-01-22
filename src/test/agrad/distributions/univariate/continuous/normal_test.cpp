#include <gtest/gtest.h>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>

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

  var p = stan::prob::normal_cdf(y,mu,sigma);

  std::vector<double> g;
  p.grad(x,g);
}
