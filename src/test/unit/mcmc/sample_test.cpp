#include <stan/mcmc/sample.hpp>
#include <vector>
#include <gtest/gtest.h>

TEST(McmcSample, size_cont) {
  
  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;
  
  double log_prob = -10;
  double accept_stat = 0.5;
  
  stan::mcmc::sample s(q, log_prob, accept_stat);
  
  EXPECT_EQ(static_cast<int>(q.size()), s.size_cont());
  
}

TEST(McmcSample, cont_params_by_index) {
  
  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;
  double log_prob = -10;
  double accept_stat = 0.5;
  
  stan::mcmc::sample s(q, log_prob, accept_stat);
  
  for (int i = 0; i < s.size_cont(); ++i)
    EXPECT_EQ(q(i), s.cont_params(i));
  
}

TEST(McmcSample, cont_params_by_vector) {
  
  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;
  double log_prob = -10;
  double accept_stat = 0.5;
  
  stan::mcmc::sample s(q, log_prob, accept_stat);
  
  const Eigen::VectorXd& q_out = s.cont_params();
  
  for (int i = 0; i < s.size_cont(); ++i)
    EXPECT_EQ(q(i), q_out(i));
  
}

TEST(McmcSample, log_prob) {
  
  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;
  double log_prob = -10;
  double accept_stat = 0.5;
  
  stan::mcmc::sample s(q, log_prob, accept_stat);
  
  EXPECT_EQ(log_prob, s.log_prob());
  
}

TEST(McmcSample, accept_stat) {
  
  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;
  double log_prob = -10;
  double accept_stat = 0.5;
  
  stan::mcmc::sample s(q, log_prob, accept_stat);
  
  EXPECT_EQ(accept_stat, s.accept_stat());
  
}
