#include <stan/mcmc/sample.hpp>
#include <vector>
#include <gtest/gtest.h>

TEST(McmcSample, size_cont) {
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  double log_prob = -10;
  double accept_stat = 0.5;
  
  stan::mcmc::sample s(q, r, log_prob, accept_stat);
  
  EXPECT_EQ(static_cast<int>(q.size()), s.size_cont());
  
}

TEST(McmcSample, cont_params_by_index) {
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  double log_prob = -10;
  double accept_stat = 0.5;
  
  stan::mcmc::sample s(q, r, log_prob, accept_stat);
  
  for (int i = 0; i < s.size_cont(); ++i)
    EXPECT_EQ(q.at(i), s.cont_params(i));
  
}

TEST(McmcSample, cont_params_by_vector) {
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  double log_prob = -10;
  double accept_stat = 0.5;
  
  stan::mcmc::sample s(q, r, log_prob, accept_stat);
  
  const std::vector<double>& q_out = s.cont_params();
  
  for (int i = 0; i < s.size_cont(); ++i)
    EXPECT_EQ(q.at(i), q_out.at(i));
  
}

TEST(McmcSample, size_disc) {
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  double log_prob = -10;
  double accept_stat = 0.5;
  
  stan::mcmc::sample s(q, r, log_prob, accept_stat);
  
  EXPECT_EQ(static_cast<int>(r.size()), s.size_disc());
  
}

TEST(McmcSample, disc_params_by_index) {
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  double log_prob = -10;
  double accept_stat = 0.5;
  
  stan::mcmc::sample s(q, r, log_prob, accept_stat);
  
  for (int i = 0; i < s.size_disc(); ++i)
    EXPECT_EQ(r.at(i), s.disc_params(i));
  
}

TEST(McmcSample, disc_params_by_vector) {
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  double log_prob = -10;
  double accept_stat = 0.5;
  
  stan::mcmc::sample s(q, r, log_prob, accept_stat);
  
  const std::vector<int>& r_out = s.disc_params();
  
  for (int i = 0; i < s.size_disc(); ++i)
    EXPECT_EQ(r.at(i), r_out.at(i));
  
}

TEST(McmcSample, log_prob) {
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  double log_prob = -10;
  double accept_stat = 0.5;
  
  stan::mcmc::sample s(q, r, log_prob, accept_stat);
  
  EXPECT_EQ(log_prob, s.log_prob());
  
}

TEST(McmcSample, accept_stat) {
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  double log_prob = -10;
  double accept_stat = 0.5;
  
  stan::mcmc::sample s(q, r, log_prob, accept_stat);
  
  EXPECT_EQ(accept_stat, s.accept_stat());
  
}
