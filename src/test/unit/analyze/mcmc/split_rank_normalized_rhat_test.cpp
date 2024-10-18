#include <stan/analyze/mcmc/compute_potential_scale_reduction.hpp>
#include <stan/analyze/mcmc/split_rank_normalized_rhat.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

TEST(RankNormalizedRhat, test_basic_bulk_tail_rhat) {
  std::stringstream out;
  Eigen::MatrixXd chains_lp(1000, 4);
  Eigen::MatrixXd chains_theta(1000, 4);

  std::vector<const double*> draws_theta(4);
  std::vector<const double*> draws_lp(4);
  std::vector<size_t> sizes(4);

  for (size_t i = 0; i < 4; ++i) {
    std::stringstream fname;
    fname << "src/test/unit/analyze/mcmc/test_csv_files/bern" << (i + 1) << ".csv"; 
    std::ifstream bern_stream(fname.str(), std::ifstream::in);
    stan::io::stan_csv bern_csv = stan::io::stan_csv_reader::parse(bern_stream, &out);
    bern_stream.close();
    chains_lp.col(i) = bern_csv.samples.col(0);
    chains_theta.col(i) = bern_csv.samples.col(7);
    draws_lp[i] = chains_lp.col(i).data();
    draws_theta[i] = chains_theta.col(i).data();
    sizes[i] = 1000;
  }
  double rhat_lp_basic_expect = 1.0001296;
  double rhat_lp_new_expect = 1.0007301;

  double rhat_theta_basic_expect = 1.0029197;
  double rhat_theta_new_expect = 1.0067897;
  
  auto rhat_basic_lp = stan::analyze::rhat(chains_lp);
  auto old_rhat_basic_lp = stan::analyze::compute_potential_scale_reduction(draws_lp, sizes);
  auto rhat_lp = stan::analyze::split_rank_normalized_rhat(chains_lp);

  auto rhat_basic_theta = stan::analyze::rhat(chains_theta);
  auto old_rhat_basic_theta = stan::analyze::compute_potential_scale_reduction(draws_theta, sizes);
  auto rhat_theta = stan::analyze::split_rank_normalized_rhat(chains_theta);

  EXPECT_NEAR(rhat_lp_basic_expect, rhat_basic_lp, 0.00001);
  EXPECT_NEAR(rhat_theta_basic_expect, rhat_basic_theta, 0.00001);

  EXPECT_NEAR(old_rhat_basic_lp, rhat_basic_lp, 0.00001);
  EXPECT_NEAR(old_rhat_basic_theta, rhat_basic_theta, 0.00001);

  EXPECT_NEAR(rhat_lp_new_expect, std::max(rhat_lp.first, rhat_lp.second), 0.00001);
  EXPECT_NEAR(rhat_theta_new_expect, std::max(rhat_theta.first, rhat_theta.second), 0.00001);
}

TEST(RankNormalizedRhat, const_fail) {
  std::stringstream out;
  std::ifstream bernoulli_const_1_stream, bernoulli_const_2_stream;
  stan::io::stan_csv bernoulli_const_1, bernoulli_const_2;
  bernoulli_const_1_stream.open(
      "src/test/unit/mcmc/test_csv_files/bernoulli_const_1.csv",
      std::ifstream::in);
  bernoulli_const_1
      = stan::io::stan_csv_reader::parse(bernoulli_const_1_stream, &out);
  bernoulli_const_1_stream.close();
  bernoulli_const_2_stream.open(
      "src/test/unit/mcmc/test_csv_files/bernoulli_const_2.csv",
      std::ifstream::in);
  bernoulli_const_2
      = stan::io::stan_csv_reader::parse(bernoulli_const_2_stream, &out);
  bernoulli_const_2_stream.close();

  Eigen::MatrixXd chains(bernoulli_const_1.samples.rows(), 2);
  chains.col(0)
      = bernoulli_const_1.samples.col(bernoulli_const_1.samples.cols() - 1);
  chains.col(1)
      = bernoulli_const_2.samples.col(bernoulli_const_2.samples.cols() - 1);
  auto rhat = stan::analyze::split_rank_normalized_rhat(chains);
  EXPECT_TRUE(std::isnan(rhat.first));
  EXPECT_TRUE(std::isnan(rhat.second));
}
