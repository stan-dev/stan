#include <stan/mcmc/stepsize_adapter.hpp>
#include <gtest/gtest.h>

TEST(McmcStepsizeAdapter, set_adapt_mu) {
  stan::mcmc::stepsize_adapter adapter;
  
  double old_mu = 0.5;
  adapter.set_adapt_mu(old_mu);
  EXPECT_EQ(old_mu, adapter.get_adapt_mu());
  
}

TEST(McmcStepsizeAdapter, set_adapt_delta) {
  stan::mcmc::stepsize_adapter adapter;
  
  double old_delta = 0.5;
  adapter.set_adapt_delta(old_delta);
  EXPECT_EQ(old_delta, adapter.get_adapt_delta());
  
  adapter.set_adapt_delta(-0.1);
  EXPECT_EQ(old_delta, adapter.get_adapt_delta());
  
  adapter.set_adapt_delta(1.1);
  EXPECT_EQ(old_delta, adapter.get_adapt_delta());
  
}

TEST(McmcStepsizeAdapter, set_adapt_gamma) {
  stan::mcmc::stepsize_adapter adapter;
  
  double old_gamma = 0.05;
  adapter.set_adapt_gamma(old_gamma);
  EXPECT_EQ(old_gamma, adapter.get_adapt_gamma());
  
  adapter.set_adapt_gamma(-0.1);
  EXPECT_EQ(old_gamma, adapter.get_adapt_gamma());
  
}

TEST(McmcStepsizeAdapter, set_adapt_kappa) {
  stan::mcmc::stepsize_adapter adapter;
  
  double old_kappa = 0.75;
  adapter.set_adapt_kappa(old_kappa);
  EXPECT_EQ(old_kappa, adapter.get_adapt_kappa());
  
  adapter.set_adapt_kappa(-0.1);
  EXPECT_EQ(old_kappa, adapter.get_adapt_kappa());
  
}

TEST(McmcStepsizeAdapter, set_adapt_t0) {
  stan::mcmc::stepsize_adapter adapter;
  
  double old_t0 = 0.05;
  adapter.set_adapt_t0(old_t0);
  EXPECT_EQ(old_t0, adapter.get_adapt_t0());
  
  adapter.set_adapt_t0(-0.1);
  EXPECT_EQ(old_t0, adapter.get_adapt_t0());
  
}

TEST(McmcStepsizeAdapter, engage_adaption) {
  stan::mcmc::stepsize_adapter adapter;
  adapter.engage_adaptation();
  EXPECT_EQ(true, adapter.adapting());
}

TEST(McmcStepsizeAdapter, disengage_adaption) {
  stan::mcmc::stepsize_adapter adapter;
  adapter.disengage_adaptation();
  EXPECT_EQ(false, adapter.adapting());
}

TEST(McmcStepsizeAdapter, learn_stepsize) {
  stan::mcmc::stepsize_adapter adapter;
  
  double target_accept = 0.65;
  double target_epsilon = 1;
  
  adapter.set_adapt_mu(log(target_epsilon));
  adapter.set_adapt_delta(target_accept);
  
  double new_epsilon = 0;
  adapter.learn_stepsize(new_epsilon, target_accept);
  
  EXPECT_EQ(target_epsilon, new_epsilon);
}