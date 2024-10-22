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

TEST_F(RankNormalizedEss, short_chains_fail) {
  chains_theta.resize(3, 4);
  auto ess = stan::analyze::split_rank_normalized_ess(chains_theta);
  EXPECT_TRUE(std::isnan(ess.first));
  EXPECT_TRUE(std::isnan(ess.second));
}
