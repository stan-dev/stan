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
    bernoulli_zeta_stream.open(
        "src/test/unit/mcmc/test_csv_files/bernoulli_zeta.csv",
        std::ifstream::in);
    eight_schools_1_stream.open(
        "src/test/unit/mcmc/test_csv_files/eight_schools_1.csv",
        std::ifstream::in);
    eight_schools_2_stream.open(
        "src/test/unit/mcmc/test_csv_files/eight_schools_2.csv",
        std::ifstream::in);
    eight_schools_5iters_1_stream.open(
        "src/test/unit/mcmc/test_csv_files/eight_schools_5iters_1.csv",
        std::ifstream::in);
    eight_schools_5iters_2_stream.open(
        "src/test/unit/mcmc/test_csv_files/eight_schools_5iters_2.csv",
        std::ifstream::in);

    if (!bernoulli_500_stream || !bernoulli_default_stream
        || !bernoulli_thin_stream || !bernoulli_warmup_stream
        || !bernoulli_zeta_stream || !eight_schools_1_stream
        || !eight_schools_2_stream || !eight_schools_5iters_1_stream
        || !eight_schools_5iters_2_stream) {
      FAIL() << "Failed to open one or more test files";
    }
    bernoulli_500_stream.seekg(0, std::ios::beg);
    bernoulli_default_stream.seekg(0, std::ios::beg);
    bernoulli_thin_stream.seekg(0, std::ios::beg);
    bernoulli_warmup_stream.seekg(0, std::ios::beg);
    bernoulli_zeta_stream.seekg(0, std::ios::beg);
    eight_schools_1_stream.seekg(0, std::ios::beg);
    eight_schools_2_stream.seekg(0, std::ios::beg);
    eight_schools_5iters_1_stream.seekg(0, std::ios::beg);
    eight_schools_5iters_2_stream.seekg(0, std::ios::beg);

    bernoulli_500
        = stan::io::stan_csv_reader::parse(bernoulli_500_stream, &out);
    bernoulli_default
        = stan::io::stan_csv_reader::parse(bernoulli_default_stream, &out);
    bernoulli_thin
        = stan::io::stan_csv_reader::parse(bernoulli_thin_stream, &out);
    bernoulli_warmup
        = stan::io::stan_csv_reader::parse(bernoulli_warmup_stream, &out);
    bernoulli_zeta
        = stan::io::stan_csv_reader::parse(bernoulli_zeta_stream, &out);
    eight_schools_1
        = stan::io::stan_csv_reader::parse(eight_schools_1_stream, &out);
    eight_schools_2
        = stan::io::stan_csv_reader::parse(eight_schools_2_stream, &out);
    eight_schools_5iters_1
        = stan::io::stan_csv_reader::parse(eight_schools_5iters_1_stream, &out);
    eight_schools_5iters_2
        = stan::io::stan_csv_reader::parse(eight_schools_5iters_2_stream, &out);
  }

  void TearDown() override {
    bernoulli_500_stream.close();
    bernoulli_default_stream.close();
    bernoulli_thin_stream.close();
    bernoulli_warmup_stream.close();
    bernoulli_zeta_stream.close();
    eight_schools_1_stream.close();
    eight_schools_2_stream.close();
    eight_schools_5iters_1_stream.close();
    eight_schools_5iters_2_stream.close();
  }

  std::stringstream out;

  std::ifstream bernoulli_500_stream, bernoulli_default_stream,
      bernoulli_thin_stream, bernoulli_warmup_stream, bernoulli_zeta_stream,
      eight_schools_1_stream, eight_schools_2_stream,
      eight_schools_5iters_1_stream, eight_schools_5iters_2_stream;

  stan::io::stan_csv bernoulli_500, bernoulli_default, bernoulli_thin,
      bernoulli_warmup, bernoulli_zeta, eight_schools_1, eight_schools_2,
      eight_schools_5iters_1, eight_schools_5iters_2;
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

  bad.clear();
  bad.push_back(bernoulli_default);
  bad.push_back(bernoulli_zeta);
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
  // mean
  // median
  // sd
  // max abs deviation
  // mcse_mean
  // mcse_sd
  // q1
  // q5
  // q95
  // q99
  // q0
  // q100
  // rhat
  // rhat_basic
  // ess_bulk, tail
  // ess_basic
  // autocovariance
}

TEST_F(McmcChains, mcse) {
  std::vector<stan::io::stan_csv> eight_schools;
  eight_schools.push_back(eight_schools_1);
  eight_schools.push_back(eight_schools_2);
  stan::mcmc::chainset chain_2(eight_schools);
  EXPECT_EQ(2, chain_2.num_chains());

  // test against R implementation in pkg posterior
  Eigen::VectorXd s8_mcse_mean(10), s8_mcse_sd(10);
  s8_mcse_mean << 0.288379, 0.4741815, 0.2741001, 0.3294614, 0.2473758,
      0.2665048, 0.2701363, 0.4740092, 0.3621771, 0.3832464;
  s8_mcse_sd << 0.1841825, 0.2854258, 0.192332, 0.2919369, 0.2478025, 0.2207478,
      0.2308452, 0.2522107, 0.2946896, 0.3184745;

  for (size_t i = 0; i < 10; ++i) {
    auto mcse_mean = chain_2.mcse_mean(i + 7);
    auto mcse_sd = chain_2.mcse_sd(i + 7);
    EXPECT_NEAR(mcse_mean, s8_mcse_mean(i), 0.05);
    EXPECT_NEAR(mcse_sd, s8_mcse_sd(i), 0.09);
  }
}

TEST_F(McmcChains, autocorrelation) {
  stan::mcmc::chainset chain_1(eight_schools_1);
  EXPECT_EQ(1, chain_1.num_chains());

  Eigen::VectorXd mu_ac_posterior(10);
  mu_ac_posterior << 1.00000000000, 0.19487668999, 0.05412049365, 0.07834048575,
      0.04145609855, 0.04353962161, -0.00977255885, 0.00005175308,
      0.01791577080, 0.01245035817;
  auto mu_ac = chain_1.autocorrelation(0, "mu");
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_NEAR(mu_ac_posterior(i), mu_ac(i), 0.0005);
  }
}
