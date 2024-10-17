#include <stan/analyze/mcmc/compute_effective_sample_size.hpp>
#include <stan/analyze/mcmc/split_rank_normalized_ess.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

TEST(RankNormalizedEss, test_basic_bulk_tail_ess) {
  std::stringstream out;
  Eigen::MatrixXd chains_lp(1000, 4);
  Eigen::MatrixXd chains_theta(1000, 4);

  std::vector<const double*> draws_theta(4);
  std::vector<const double*> draws_lp(4);
  std::vector<size_t> sizes(4);

  for (size_t i = 0; i < 4; ++i) {
    std::stringstream fname;
    fname << "src/test/unit/analyze/mcmc/test_csv_files/bern" << (i + 1)
          << ".csv";
    std::ifstream bern_stream(fname.str(), std::ifstream::in);
    stan::io::stan_csv bern_csv
        = stan::io::stan_csv_reader::parse(bern_stream, &out);
    bern_stream.close();
    chains_lp.col(i) = bern_csv.samples.col(0);
    chains_theta.col(i) = bern_csv.samples.col(7);
    draws_lp[i] = chains_lp.col(i).data();
    draws_theta[i] = chains_theta.col(i).data();
    sizes[i] = 1000;
  }
  double ess_lp_expect = 1335.4137;
  double ess_lp_bulk_expect = 1512.7684;
  double ess_lp_tail_expect = 1591.9707;

  double ess_theta_expect = 1377.503;
  double ess_theta_bulk_expect = 1407.5124;
  double ess_theta_tail_expect = 1291.7131;

  auto ess_basic_lp = stan::analyze::ess(chains_lp);
  auto old_ess_basic_lp
      = stan::analyze::compute_effective_sample_size(draws_lp, sizes);
  auto ess_lp = stan::analyze::split_rank_normalized_ess(chains_lp);

  auto ess_basic_theta = stan::analyze::ess(chains_theta);
  auto old_ess_basic_theta
      = stan::analyze::compute_effective_sample_size(draws_theta, sizes);
  auto ess_theta = stan::analyze::split_rank_normalized_ess(chains_theta);

  EXPECT_NEAR(ess_lp_expect, ess_basic_lp, 0.001);
  EXPECT_NEAR(ess_theta_expect, ess_basic_theta, 0.001);

  EXPECT_NEAR(old_ess_basic_lp, ess_basic_lp, 0.00001);
  EXPECT_NEAR(old_ess_basic_theta, ess_basic_theta, 0.00001);

  EXPECT_NEAR(ess_lp_bulk_expect, ess_lp.first, 0.001);
  EXPECT_NEAR(ess_lp_tail_expect, ess_lp.second, 0.001);

  EXPECT_NEAR(ess_theta_bulk_expect, ess_theta.first, 0.001);
  EXPECT_NEAR(ess_theta_tail_expect, ess_theta.second, 0.001);
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
