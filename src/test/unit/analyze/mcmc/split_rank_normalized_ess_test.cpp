#include <stan/analyze/mcmc/split_rank_normalized_ess.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

class RankNormalizedEss : public testing::Test {
 public:
  void SetUp() {
    chains_lp.resize(1000, 4);
    chains_theta.resize(1000, 4);
    chains_divergent.resize(1000, 4);
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
      chains_divergent.col(i) = bern_csv.samples.col(5);
    }
  }

  void TearDown() {}

  std::stringstream out;
  Eigen::MatrixXd chains_lp;
  Eigen::MatrixXd chains_theta;
  Eigen::MatrixXd chains_divergent;
};

TEST_F(RankNormalizedEss, test_bulk_tail_ess) {
  // computed via R pkg posterior
  double ess_lp_bulk_expect = 1512.7684;
  double ess_lp_tail_expect = 1591.9707;

  double ess_theta_bulk_expect = 1407.5124;
  double ess_theta_tail_expect = 1291.7131;

  auto ess_lp = stan::analyze::split_rank_normalized_ess(chains_lp);
  auto ess_theta = stan::analyze::split_rank_normalized_ess(chains_theta);

  EXPECT_NEAR(ess_lp_bulk_expect, ess_lp.first, 1e-4);
  EXPECT_NEAR(ess_lp_tail_expect, ess_lp.second, 1e-4);

  EXPECT_NEAR(ess_theta_bulk_expect, ess_theta.first, 1e-4);
  EXPECT_NEAR(ess_theta_tail_expect, ess_theta.second, 1e-4);
}

TEST_F(RankNormalizedEss, const_fail) {
  auto ess = stan::analyze::split_rank_normalized_ess(chains_divergent);
  EXPECT_TRUE(std::isnan(ess.first));
  EXPECT_TRUE(std::isnan(ess.second));
}

TEST_F(RankNormalizedEss, inf_fail) {
  chains_theta(0, 0) = std::numeric_limits<double>::infinity();
  auto ess = stan::analyze::split_rank_normalized_ess(chains_theta);
  EXPECT_TRUE(std::isnan(ess.first));
  EXPECT_TRUE(std::isnan(ess.second));
}

TEST_F(RankNormalizedEss, constant_chain) {
  // computed via R pkg posterior
  chains_theta.col(3).setConstant(0.3);
  auto ess = stan::analyze::split_rank_normalized_ess(chains_theta);
  EXPECT_NEAR(794.887967121017, ess.first, 1e-6);
  EXPECT_NEAR(1194.06701581485, ess.second, 1e-6);
}

TEST_F(RankNormalizedEss, half_constant_chain) {
  // computed via R pkg posterior
  chains_theta.col(3).tail(500).setConstant(0.3);
  auto ess = stan::analyze::split_rank_normalized_ess(chains_theta);
  EXPECT_NEAR(1108.06541725348, ess.first, 1e-6);
  EXPECT_NEAR(1288.5051347075, ess.second, 1e-6);
}

TEST_F(RankNormalizedEss, chain_outside_both_tails) {
  // computed via R pkg posterior
  // chain 4 is above the 5% quantile, and the other chains are below the
  // 95% quantile
  chains_theta.col(3).array() += 1.0;
  auto ess = stan::analyze::split_rank_normalized_ess(chains_theta);
  EXPECT_NEAR(7.24636932115751, ess.first, 1e-8);
  EXPECT_NEAR(32.1273579765829, ess.second, 1e-8);
}

TEST_F(RankNormalizedEss, every_chain_constant) {
  Eigen::MatrixXd chains(1000, 4);
  for (int i = 0; i < 4; ++i) {
    chains.col(i).setConstant(i + 1);
  }
  auto ess = stan::analyze::split_rank_normalized_ess(chains);
  EXPECT_TRUE(std::isnan(ess.first));
  EXPECT_TRUE(std::isnan(ess.second));
}

TEST_F(RankNormalizedEss, short_chains_fail) {
  chains_theta.resize(3, 4);
  auto ess = stan::analyze::split_rank_normalized_ess(chains_theta);
  EXPECT_TRUE(std::isnan(ess.first));
  EXPECT_TRUE(std::isnan(ess.second));
}
