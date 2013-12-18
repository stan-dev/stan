#include <stan/agrad/rev.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <stan/agrad/fwd/prob/univariate/continuous/normal.hpp>
#include <gtest/gtest.h>
#include <vector>


std::vector<double> test_fun(double y, double mu, double sigma) {
  using stan::agrad::var;
  using stan::prob::normal_log;
  var y_var = y;
  var mu_var = mu;
  var sigma_var = sigma;
  
  std::vector<var> x;
  x.push_back(y_var);
  x.push_back(mu_var);
  x.push_back(sigma_var);

  var logp = normal_log<false>(y_var,mu_var,sigma_var);
  std::vector<double> grad;
  logp.grad(x,grad);
  return grad;
}

TEST(ProbAgradDistributionsNormal, fwd) {
  using stan::agrad::fvar;
  using stan::prob::normal_log;

  EXPECT_FLOAT_EQ(-0.918938533204673, 
                  normal_log<false>(0,0,1));
  EXPECT_FLOAT_EQ(-0.918938533204673, 
                  normal_log<false>(0,0,fvar<double>(1.0)).val());
  EXPECT_FLOAT_EQ(-0.918938533204673, 
                  normal_log<false>(0,fvar<double>(0),1).val());
  EXPECT_FLOAT_EQ(-0.918938533204673, 
                  normal_log<false>(0,fvar<double>(0),fvar<double>(1)).val());
  EXPECT_FLOAT_EQ(-0.918938533204673, 
                  normal_log<false>(fvar<double>(0),0,1).val());
  EXPECT_FLOAT_EQ(-0.918938533204673, 
                  normal_log<false>(fvar<double>(0),0,fvar<double>(1)).val());
  EXPECT_FLOAT_EQ(-0.918938533204673, 
                  normal_log<false>(fvar<double>(0),fvar<double>(0),1).val());
  EXPECT_FLOAT_EQ(-0.918938533204673, 
                  normal_log<false>(fvar<double>(0),fvar<double>(0),fvar<double>(1)).val());
}

TEST(ProbAgradDistributionsNormal, derivatives) {
  using stan::agrad::fvar;
  using stan::prob::normal_log;
   
  std::vector<double> grad = test_fun(0,0,1);

  fvar<double> lp = normal_log<false>(0,0,fvar<double>(1.0,1));
  EXPECT_FLOAT_EQ(grad[2], lp.tangent());
}
