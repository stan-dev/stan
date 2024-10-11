#include <stan/analyze/mcmc/split_rank_normalized_rhat.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

TEST(RankNormalizedRhat, compute_split_rank_normalized_rhat) {
  std::stringstream out;
  std::ifstream eight_schools_1_stream;
  stan::io::stan_csv eight_schools_1;
  eight_schools_1_stream.open(
      "src/test/unit/mcmc/test_csv_files/eight_schools_1.csv",
      std::ifstream::in);
  eight_schools_1
      = stan::io::stan_csv_reader::parse(eight_schools_1_stream, &out);
  eight_schools_1_stream.close();

  // test against R implementation in pkg posterior
  Eigen::VectorXd rhat_8_schools_1_bulk(10);
  rhat_8_schools_1_bulk << 1.0012958313, 1.0046136496, 1.0085723580,
      1.0248629375, 1.0111456620, 1.0004458336, 0.9987162973, 1.0339773469,
      0.9985612618, 1.0281667351;

  Eigen::VectorXd rhat_8_schools_1_tail(10);
  rhat_8_schools_1_tail << 1.005676523, 1.009670999, 1.00184184, 1.002222679,
      1.004148161, 1.003218528, 1.009195353, 1.001426744, 1.003984381,
      1.025817745;

  Eigen::MatrixXd chains(eight_schools_1.samples.rows(), 1);
  for (size_t i = 0; i < 10; ++i) {
    chains.col(0) = eight_schools_1.samples.col(i + 7);
    auto rhats = stan::analyze::split_rank_normalized_rhat(chains);
    EXPECT_NEAR(rhats.first, rhat_8_schools_1_bulk(i), 0.05);
    EXPECT_NEAR(rhats.second, rhat_8_schools_1_tail(i), 0.05);
  }
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
