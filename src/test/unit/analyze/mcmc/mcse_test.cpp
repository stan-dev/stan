#include <stan/analyze/mcmc/mcse.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

class MonteCarloStandardError : public testing::Test {
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

TEST_F(MonteCarloStandardError, test_mcse) {
  // computed via R pkg posterior
  double mcse_mean_lp_expect = 0.02016;
  double mcse_mean_theta_expect = 0.00323;

  double mcse_sd_lp_expect = 0.03553;
  double mcse_sd_theta_expect = 0.00216;
  EXPECT_NEAR(mcse_mean_lp_expect, stan::analyze::mcse_mean(chains_lp), 1e-4);
  EXPECT_NEAR(mcse_mean_theta_expect, stan::analyze::mcse_mean(chains_theta),
              1e-4);

  EXPECT_NEAR(mcse_sd_lp_expect, stan::analyze::mcse_sd(chains_lp), 1e-4);
  EXPECT_NEAR(mcse_sd_theta_expect, stan::analyze::mcse_sd(chains_theta), 1e-4);
}

TEST_F(MonteCarloStandardError, const_fail) {
  auto mcse_mean = stan::analyze::mcse_mean(chains_divergent);
  auto mcse_sd = stan::analyze::mcse_sd(chains_divergent);
  EXPECT_TRUE(std::isnan(mcse_mean));
  EXPECT_TRUE(std::isnan(mcse_sd));
}

TEST_F(MonteCarloStandardError, inf_fail) {
  chains_theta(0, 0) = std::numeric_limits<double>::infinity();
  auto mcse_mean = stan::analyze::mcse_mean(chains_theta);
  auto mcse_sd = stan::analyze::mcse_sd(chains_theta);
  EXPECT_TRUE(std::isnan(mcse_mean));
  EXPECT_TRUE(std::isnan(mcse_sd));
}

TEST_F(MonteCarloStandardError, short_chains_fail) {
  chains_theta.resize(3, 4);
  auto mcse_mean = stan::analyze::mcse_mean(chains_theta);
  auto mcse_sd = stan::analyze::mcse_sd(chains_theta);
  EXPECT_TRUE(std::isnan(mcse_mean));
  EXPECT_TRUE(std::isnan(mcse_sd));
}
