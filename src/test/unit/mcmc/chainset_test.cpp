#include <stan/mcmc/chainset.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <set>
#include <exception>
#include <utility>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

class McmcChains : public testing::Test {
 public:
  void SetUp() override {
    bernoulli_500_stream.open(
        "src/test/unit/mcmc/test_csv_files/bernoulli_500.csv",
        std::ifstream::in);
    bernoulli_default_stream.open(
        "src/test/unit/mcmc/test_csv_files/bernoulli_default.csv",
        std::ifstream::in);
    bernoulli_thin_stream.open(
        "src/test/unit/mcmc/test_csv_files/bernoulli_thin.csv",
        std::ifstream::in);
    bernoulli_warmup_stream.open(
        "src/test/unit/mcmc/test_csv_files/bernoulli_warmup.csv",
        std::ifstream::in);
    eight_schools_1_stream.open(
        "src/test/unit/mcmc/test_csv_files/eight_schools_1.csv",
        std::ifstream::in);
    eight_schools_2_stream.open(
        "src/test/unit/mcmc/test_csv_files/eight_schools_2.csv",
        std::ifstream::in);

    if (!bernoulli_500_stream || !bernoulli_default_stream
        || !bernoulli_thin_stream || !bernoulli_warmup_stream
        || !eight_schools_1_stream || !eight_schools_2_stream) {
      FAIL() << "Failed to open one or more test files";
    }
    bernoulli_500_stream.seekg(0, std::ios::beg);
    bernoulli_default_stream.seekg(0, std::ios::beg);
    bernoulli_thin_stream.seekg(0, std::ios::beg);
    bernoulli_warmup_stream.seekg(0, std::ios::beg);
    eight_schools_1_stream.seekg(0, std::ios::beg);
    eight_schools_2_stream.seekg(0, std::ios::beg);

    bernoulli_500
        = stan::io::stan_csv_reader::parse(bernoulli_500_stream, &out);
    bernoulli_default
        = stan::io::stan_csv_reader::parse(bernoulli_default_stream, &out);
    bernoulli_thin
        = stan::io::stan_csv_reader::parse(bernoulli_thin_stream, &out);
    bernoulli_warmup
        = stan::io::stan_csv_reader::parse(bernoulli_warmup_stream, &out);
    eight_schools_1
        = stan::io::stan_csv_reader::parse(eight_schools_1_stream, &out);
    eight_schools_2
        = stan::io::stan_csv_reader::parse(eight_schools_2_stream, &out);
  }

  void TearDown() override {
    bernoulli_500_stream.close();
    bernoulli_default_stream.close();
    bernoulli_thin_stream.close();
    bernoulli_warmup_stream.close();
    eight_schools_1_stream.close();
    eight_schools_2_stream.close();
  }

  std::stringstream out;

  std::ifstream bernoulli_500_stream, bernoulli_default_stream,
      bernoulli_thin_stream, bernoulli_warmup_stream, eight_schools_1_stream,
      eight_schools_2_stream;

  stan::io::stan_csv bernoulli_500, bernoulli_default, bernoulli_thin,
      bernoulli_warmup, eight_schools_1, eight_schools_2;
};

TEST_F(McmcChains, constructor) {
  stan::mcmc::chainset chain_1(eight_schools_1);
  EXPECT_EQ(1, chain_1.num_chains());
  EXPECT_EQ(eight_schools_1.header.size(), chain_1.num_params());
  EXPECT_EQ(
      eight_schools_1.metadata.num_samples / eight_schools_1.metadata.thin,
      chain_1.num_samples());

  std::vector<stan::io::stan_csv> eight_schools;
  eight_schools.push_back(eight_schools_1);
  eight_schools.push_back(eight_schools_2);
  stan::mcmc::chainset chain_2(eight_schools);
  EXPECT_EQ(2, chain_2.num_chains());
  EXPECT_EQ(eight_schools_1.header.size(), chain_2.num_params());
  EXPECT_EQ(
      eight_schools_1.metadata.num_samples / eight_schools_1.metadata.thin,
      chain_2.num_samples());

  std::vector<stan::io::stan_csv> bernoulli;
  bernoulli.push_back(bernoulli_default);
  bernoulli.push_back(bernoulli_thin);
  bernoulli.push_back(bernoulli_warmup);
  stan::mcmc::chainset chain_3(bernoulli);
  EXPECT_EQ(3, chain_3.num_chains());
  EXPECT_EQ(bernoulli_default.header.size(), chain_3.num_params());
  EXPECT_EQ(bernoulli_default.metadata.num_samples, chain_3.num_samples());
}

TEST_F(McmcChains, addFail) {
  std::vector<stan::io::stan_csv> bad;
  bad.push_back(bernoulli_default);
  bad.push_back(bernoulli_500);
  EXPECT_THROW(stan::mcmc::chainset fail(bad), std::invalid_argument);

  bad.clear();
  bad.push_back(bernoulli_default);
  bad.push_back(eight_schools_1);
  EXPECT_THROW(stan::mcmc::chainset fail(bad), std::invalid_argument);
}

TEST_F(McmcChains, paramNameIndex) {
  stan::mcmc::chainset chains_csv(eight_schools_1);
  EXPECT_EQ(1, chains_csv.num_chains());
  for (int i = 0; i < eight_schools_1.header.size(); i++) {
    EXPECT_EQ(eight_schools_1.header[i], chains_csv.param_name(i));
    EXPECT_EQ(i, chains_csv.index(eight_schools_1.header[i]));
  }
  EXPECT_THROW(chains_csv.param_name(-5000), std::invalid_argument);
  EXPECT_THROW(chains_csv.param_name(5000), std::invalid_argument);
  EXPECT_THROW(chains_csv.index("foo"), std::invalid_argument);
}

TEST_F(McmcChains, eight_schools_samples) {
  std::vector<stan::io::stan_csv> eight_schools;
  eight_schools.push_back(eight_schools_1);
  eight_schools.push_back(eight_schools_2);
  stan::mcmc::chainset chain_2(eight_schools);
  Eigen::MatrixXd mu_all = chain_2.samples("mu");
  EXPECT_EQ(chain_2.num_chains() * chain_2.num_samples(), mu_all.size());
  mu_all = chain_2.samples(7);
  EXPECT_EQ(chain_2.num_chains() * chain_2.num_samples(), mu_all.size());

  EXPECT_THROW(chain_2.samples(5000), std::invalid_argument);
  EXPECT_THROW(chain_2.samples("foo"), std::invalid_argument);
}

TEST_F(McmcChains, summary_stats) {
  std::stringstream out;
  std::vector<stan::io::stan_csv> bern_csvs(4);
  for (size_t i = 0; i < 4; ++i) {
    std::stringstream fname;
    fname << "src/test/unit/analyze/mcmc/test_csv_files/bern" << (i + 1)
          << ".csv";
    std::ifstream bern_stream(fname.str(), std::ifstream::in);
    stan::io::stan_csv bern_csv
        = stan::io::stan_csv_reader::parse(bern_stream, &out);
    bern_stream.close();
    bern_csvs[i] = bern_csv;
  }
  stan::mcmc::chainset bern_chains(bern_csvs);
  EXPECT_EQ(4, bern_chains.num_chains());

  Eigen::MatrixXd theta = bern_chains.samples("theta");
  // default summary statistics - via R pkg posterior
  double theta_mean_expect = 0.251297;
  EXPECT_NEAR(theta_mean_expect, bern_chains.mean("theta"), 1e-5);

  double theta_median_expect = 0.237476;
  EXPECT_NEAR(theta_median_expect, bern_chains.median("theta"), 1e-5);

  double theta_sd_expect = 0.121546;
  EXPECT_NEAR(theta_sd_expect, bern_chains.sd("theta"), 1e-5);

  double theta_mad_expect = 0.12309;
  EXPECT_NEAR(theta_mad_expect, bern_chains.max_abs_deviation("theta"), 1e-5);

  double theta_mcse_mean_expect = 0.003234;
  EXPECT_NEAR(theta_mcse_mean_expect, bern_chains.mcse_mean("theta"), 1e-4);

  double theta_mcse_sd_expect = 0.002164;
  EXPECT_NEAR(theta_mcse_sd_expect, bern_chains.mcse_sd("theta"), 1e-4);

  Eigen::VectorXd probs(6);
  probs << 0.0, 0.01, 0.05, 0.95, 0.99, 1.0;
  Eigen::VectorXd quantiles_expect(6);
  quantiles_expect << 0.004072, 0.046281, 0.077169, 0.473885, 0.574524,
      0.698401;
  Eigen::VectorXd theta_quantiles = bern_chains.quantiles("theta", probs);
  for (size_t i = 0; i < probs.size(); ++i) {
    EXPECT_NEAR(quantiles_expect(i), theta_quantiles(i), 1e-5);
  }

  double theta_rhat_expect = 1.00679;
  auto rhat = bern_chains.split_rank_normalized_rhat("theta");
  EXPECT_NEAR(theta_rhat_expect, std::max(rhat.first, rhat.second), 1e-5);

  double theta_ess_bulk_expect = 1407.5124;
  double theta_ess_tail_expect = 1291.7131;
  auto ess = bern_chains.split_rank_normalized_ess("theta");
  EXPECT_NEAR(theta_ess_bulk_expect, ess.first, 1e-4);
  EXPECT_NEAR(theta_ess_tail_expect, ess.second, 1e-4);

  // autocorrelation - first 10 lags
  Eigen::VectorXd theta_ac_expect(10);
  theta_ac_expect << 1.00000, 0.42204, 0.20683, 0.08383, 0.037326, 0.02507,
      0.02003, 0.01347, 0.00476, 0.029495;
  auto theta_ac = bern_chains.autocorrelation(0, "theta");
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_NEAR(theta_ac(i), theta_ac_expect(i), 0.0005);
  }
}
