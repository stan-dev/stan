#include <stan/analyze/mcmc/split_rank_normalized_rhat.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <limits>

class RankNormalizedRhat : public testing::Test {
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

TEST_F(RankNormalizedRhat, test_bulk_tail_rhat) {
  // computed via R pkg posterior
  double rhat_lp_expect = 1.00073;
  double rhat_theta_expect = 1.006789;

  auto rhat_lp = stan::analyze::split_rank_normalized_rhat(chains_lp);
  auto rhat_theta = stan::analyze::split_rank_normalized_rhat(chains_theta);

  EXPECT_NEAR(rhat_lp_expect, std::max(rhat_lp.first, rhat_lp.second), 1e-5);
  EXPECT_NEAR(rhat_theta_expect, std::max(rhat_theta.first, rhat_theta.second),
              1e-5);
}

TEST_F(RankNormalizedRhat, const_fail) {
  auto rhat = stan::analyze::split_rank_normalized_rhat(chains_divergent);
  EXPECT_TRUE(std::isnan(rhat.first));
  EXPECT_TRUE(std::isnan(rhat.second));
}

TEST_F(RankNormalizedRhat, inf_fail) {
  chains_theta(0, 0) = std::numeric_limits<double>::infinity();
  auto rhat = stan::analyze::split_rank_normalized_rhat(chains_theta);
  EXPECT_TRUE(std::isnan(rhat.first));
  EXPECT_TRUE(std::isnan(rhat.second));
}
