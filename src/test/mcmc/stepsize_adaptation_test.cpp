#include <stan/mcmc/stepsize_adaptation.hpp>
#include <gtest/gtest.h>

TEST(McmcStepsizeAdaptation, set_mu) {
  stan::mcmc::stepsize_adaptation adaptation;
  
  double old_mu = 0.5;
  adaptation.set_mu(old_mu);
  EXPECT_EQ(old_mu, adaptation.get_mu());
  
}

TEST(McmcStepsizeAdaptation, set_delta) {
  stan::mcmc::stepsize_adaptation adaptation;
  
  double old_delta = 0.5;
  adaptation.set_delta(old_delta);
  EXPECT_EQ(old_delta, adaptation.get_delta());
  
  adaptation.set_delta(-0.1);
  EXPECT_EQ(old_delta, adaptation.get_delta());
  
  adaptation.set_delta(1.1);
  EXPECT_EQ(old_delta, adaptation.get_delta());
  
}

TEST(McmcStepsizeAdaptation, set_gamma) {
  stan::mcmc::stepsize_adaptation adaptation;
  
  double old_gamma = 0.05;
  adaptation.set_gamma(old_gamma);
  EXPECT_EQ(old_gamma, adaptation.get_gamma());
  
  adaptation.set_gamma(-0.1);
  EXPECT_EQ(old_gamma, adaptation.get_gamma());
  
}

TEST(McmcStepsizeAdaptation, set_kappa) {
  stan::mcmc::stepsize_adaptation adaptation;
  
  double old_kappa = 0.75;
  adaptation.set_kappa(old_kappa);
  EXPECT_EQ(old_kappa, adaptation.get_kappa());
  
  adaptation.set_kappa(-0.1);
  EXPECT_EQ(old_kappa, adaptation.get_kappa());
  
}

TEST(McmcStepsizeAdaptation, set_t0) {
  stan::mcmc::stepsize_adaptation adaptation;
  
  double old_t0 = 0.05;
  adaptation.set_t0(old_t0);
  EXPECT_EQ(old_t0, adaptation.get_t0());
  
  adaptation.set_t0(-0.1);
  EXPECT_EQ(old_t0, adaptation.get_t0());
  
}

TEST(McmcStepsizeAdaptation, learn_stepsize) {
  stan::mcmc::stepsize_adaptation adaptation;
  
  double target_accept = 0.65;
  double target_epsilon = 1;
  
  adaptation.set_mu(log(target_epsilon));
  adaptation.set_delta(target_accept);
  
  double new_epsilon = 0;
  adaptation.learn_stepsize(new_epsilon, target_accept);
  
  EXPECT_EQ(target_epsilon, new_epsilon);
}