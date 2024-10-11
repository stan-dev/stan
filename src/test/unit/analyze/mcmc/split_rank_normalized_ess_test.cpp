#include <stan/analyze/mcmc/split_rank_normalized_ess.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

TEST(RankNormalizedEss, compute_split_rank_normalized_ess) {
  std::stringstream out;
  std::ifstream eight_schools_1_stream, eight_schools_2_stream;
  stan::io::stan_csv eight_schools_1, eight_schools_2;
  eight_schools_1_stream.open(
      "src/test/unit/mcmc/test_csv_files/eight_schools_1.csv",
      std::ifstream::in);
  eight_schools_1
      = stan::io::stan_csv_reader::parse(eight_schools_1_stream, &out);
  eight_schools_1_stream.close();

  eight_schools_2_stream.open(
      "src/test/unit/mcmc/test_csv_files/eight_schools_2.csv",
      std::ifstream::in);
  eight_schools_2
      = stan::io::stan_csv_reader::parse(eight_schools_2_stream, &out);
  eight_schools_2_stream.close();

  // test against R implementation in pkg posterior (via cmdstanr)
  Eigen::VectorXd ess_8_schools_bulk(10);
  ess_8_schools_bulk << 348, 370, 600, 638, 765, 608, 629, 274, 517, 112;
  Eigen::VectorXd ess_8_schools_tail(10);
  ess_8_schools_tail << 845, 858, 874, 726, 620, 753, 826, 628, 587, 108;

  Eigen::MatrixXd chains(eight_schools_1.samples.rows(), 2);
  for (size_t i = 0; i < 10; ++i) {
    chains.col(0) = eight_schools_1.samples.col(i + 7);
    chains.col(1) = eight_schools_2.samples.col(i + 7);
    auto ess = stan::analyze::split_rank_normalized_ess(chains);
    EXPECT_NEAR(ess.first, ess_8_schools_bulk(i), 5);
    EXPECT_NEAR(ess.second, ess_8_schools_tail(i), 5);
  }
}

TEST(RankNormalizedEss, short_chains_fail) {
  std::stringstream out;
  std::ifstream eight_schools_5iters_1_stream, eight_schools_5iters_2_stream;
  stan::io::stan_csv eight_schools_5iters_1, eight_schools_5iters_2;
  eight_schools_5iters_1_stream.open(
      "src/test/unit/mcmc/test_csv_files/eight_schools_5iters_1.csv",
      std::ifstream::in);
  eight_schools_5iters_1
      = stan::io::stan_csv_reader::parse(eight_schools_5iters_1_stream, &out);
  eight_schools_5iters_1_stream.close();
  eight_schools_5iters_2_stream.open(
      "src/test/unit/mcmc/test_csv_files/eight_schools_5iters_2.csv",
      std::ifstream::in);
  eight_schools_5iters_2
      = stan::io::stan_csv_reader::parse(eight_schools_5iters_2_stream, &out);
  eight_schools_5iters_2_stream.close();

  Eigen::MatrixXd chains(eight_schools_5iters_1.samples.rows(), 2);
  for (size_t i = 0; i < 10; ++i) {
    chains.col(0) = eight_schools_5iters_1.samples.col(i + 7);
    chains.col(1) = eight_schools_5iters_2.samples.col(i + 7);
    auto ess = stan::analyze::split_rank_normalized_ess(chains);
    EXPECT_TRUE(std::isnan(ess.first));
    EXPECT_TRUE(std::isnan(ess.second));
  }
}
