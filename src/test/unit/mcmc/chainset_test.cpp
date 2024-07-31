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
    eight_schools_1_stream.open("src/test/unit/mcmc/test_csv_files/eight_schools_1.csv", std::ifstream::in);
    eight_schools_2_stream.open("src/test/unit/mcmc/test_csv_files/eight_schools_2.csv", std::ifstream::in);
    bernoulli_500_stream.open("src/test/unit/mcmc/test_csv_files/bernoulli_500.csv", std::ifstream::in);
    bernoulli_corrupt_stream.open("src/test/unit/mcmc/test_csv_files/bernoulli_corrupt.csv", std::ifstream::in);
    bernoulli_default_stream.open("src/test/unit/mcmc/test_csv_files/bernoulli_default.csv", std::ifstream::in);
    bernoulli_thin_stream.open("src/test/unit/mcmc/test_csv_files/bernoulli_thin.csv", std::ifstream::in);
    bernoulli_warmup_stream.open("src/test/unit/mcmc/test_csv_files/bernoulli_warmup.csv", std::ifstream::in);
    bernoulli_zeta_stream.open("src/test/unit/mcmc/test_csv_files/bernoulli_zeta.csv", std::ifstream::in);

    if (!eight_schools_1_stream || !eight_schools_2_stream
	|| !bernoulli_500_stream || !bernoulli_corrupt_stream || !bernoulli_default_stream
	|| !bernoulli_thin_stream || !bernoulli_warmup_stream || !bernoulli_zeta_stream) {
      FAIL() << "Failed to open one or more test files";
    }
    eight_schools_1_stream.seekg(0, std::ios::beg);
    eight_schools_2_stream.seekg(0, std::ios::beg);
    bernoulli_500_stream.seekg(0, std::ios::beg);
    bernoulli_corrupt_stream.seekg(0, std::ios::beg);
    bernoulli_default_stream.seekg(0, std::ios::beg);
    bernoulli_thin_stream.seekg(0, std::ios::beg);
    bernoulli_warmup_stream.seekg(0, std::ios::beg);
    bernoulli_zeta_stream.seekg(0, std::ios::beg);
    eight_schools_1 = stan::io::stan_csv_reader::parse(eight_schools_1_stream, &out);
    eight_schools_2 = stan::io::stan_csv_reader::parse(eight_schools_2_stream, &out);
    bernoulli_500 = stan::io::stan_csv_reader::parse(bernoulli_500_stream, &out);
    bernoulli_corrupt = stan::io::stan_csv_reader::parse(bernoulli_corrupt_stream, &out);
    bernoulli_default = stan::io::stan_csv_reader::parse(bernoulli_default_stream, &out);
    bernoulli_thin = stan::io::stan_csv_reader::parse(bernoulli_thin_stream, &out);
    bernoulli_warmup = stan::io::stan_csv_reader::parse(bernoulli_warmup_stream, &out);
    bernoulli_zeta = stan::io::stan_csv_reader::parse(bernoulli_zeta_stream, &out);
  }

  void TearDown() override {
    eight_schools_1_stream.close();
    eight_schools_2_stream.close();
    bernoulli_500_stream.close();
    bernoulli_corrupt_stream.close();
    bernoulli_default_stream.close();
    bernoulli_thin_stream.close();
    bernoulli_warmup_stream.close();
    bernoulli_zeta_stream.close();
  }

  std::stringstream out;
  std::ifstream eight_schools_1_stream, eight_schools_2_stream,
    bernoulli_500_stream,
    bernoulli_corrupt_stream,  bernoulli_default_stream,
    bernoulli_thin_stream, bernoulli_warmup_stream, bernoulli_zeta_stream;
  stan::io::stan_csv eight_schools_1, eight_schools_2,
    bernoulli_500, bernoulli_corrupt,
    bernoulli_default, bernoulli_thin, bernoulli_warmup, bernoulli_zeta;
};

TEST_F(McmcChains, constructor) {
  stan::mcmc::chainset<> chain_1(eight_schools_1);
  EXPECT_EQ(1, chain_1.num_chains());
  EXPECT_EQ(eight_schools_1.header.size(), chain_1.num_params());
  EXPECT_EQ(eight_schools_1.metadata.num_samples/eight_schools_1.metadata.thin,
	    chain_1.num_samples());

  std::vector<stan::io::stan_csv> eight_schools;
  eight_schools.push_back(eight_schools_1);
  eight_schools.push_back(eight_schools_2);
  stan::mcmc::chainset<> chain_2(eight_schools);
  EXPECT_EQ(2, chain_2.num_chains());
  EXPECT_EQ(eight_schools_1.header.size(), chain_2.num_params());
  EXPECT_EQ(eight_schools_1.metadata.num_samples/eight_schools_1.metadata.thin,
	    chain_2.num_samples());

  std::vector<stan::io::stan_csv> bernoulli;
  bernoulli.push_back(bernoulli_default);
  bernoulli.push_back(bernoulli_thin);
  bernoulli.push_back(bernoulli_warmup);
  stan::mcmc::chainset<> chain_3(bernoulli);
  EXPECT_EQ(3, chain_3.num_chains());
  EXPECT_EQ(bernoulli_default.header.size(), chain_3.num_params());
  EXPECT_EQ(bernoulli_default.metadata.num_samples, chain_3.num_samples());
}

TEST_F(McmcChains, addFail) {
  std::vector<stan::io::stan_csv> bad;
  bad.push_back(bernoulli_default);
  bad.push_back(bernoulli_corrupt);
  EXPECT_THROW(stan::mcmc::chainset<> fail(bad), std::invalid_argument);

  bad.clear();
  bad.push_back(bernoulli_default);
  bad.push_back(bernoulli_500);
  EXPECT_THROW(stan::mcmc::chainset<> fail(bad), std::invalid_argument);

  bad.clear();
  bad.push_back(bernoulli_default);
  bad.push_back(eight_schools_1);
  EXPECT_THROW(stan::mcmc::chainset<> fail(bad), std::invalid_argument);

  bad.clear();
  bad.push_back(bernoulli_default);
  bad.push_back(bernoulli_zeta);
  EXPECT_THROW(stan::mcmc::chainset<> fail(bad), std::invalid_argument);
}

TEST_F(McmcChains, paramNameIndex) {
  stan::mcmc::chainset<> chains_csv(eight_schools_1);
  EXPECT_EQ(1, chains_csv.num_chains());
  for (int i = 0; i < eight_schools_1.header.size(); i++) {
    EXPECT_EQ(eight_schools_1.header[i], chains_csv.param_name(i));
    EXPECT_EQ(i, chains_csv.index(eight_schools_1.header[i]));
  }
}

TEST_F(McmcChains, eight_schools_samples) {
  std::vector<stan::io::stan_csv> eight_schools;
  eight_schools.push_back(eight_schools_1);
  eight_schools.push_back(eight_schools_2);
  stan::mcmc::chainset<> chain_2(eight_schools);

  Eigen::VectorXd mu_1 = chain_2.samples(0, "mu");
  Eigen::VectorXd mu_2 = chain_2.samples(1, "mu");
  EXPECT_EQ(mu_1.size(), mu_2.size());
  Eigen::VectorXd mu_all = chain_2.samples("mu");
  EXPECT_EQ(mu_1.size() + mu_2.size(), mu_all.size());
}

TEST_F(McmcChains, split_rank_normalized_rhat) {
  stan::mcmc::chainset<> chain_1(eight_schools_1);
  EXPECT_EQ(1, chain_1.num_chains());

  // test against R implementation in pkg posterior
  Eigen::VectorXd rhat_8_schools_1_bulk(10);
  rhat_8_schools_1_bulk <<
    1.0012958313, 1.0046136496, 1.0085723580, 1.0248629375,
    1.0111456620, 1.0004458336, 0.9987162973, 1.0339773469,
    0.9985612618, 1.0281667351;

  Eigen::VectorXd rhat_8_schools_1_tail(10);
  rhat_8_schools_1_tail <<
    1.005676523, 1.009670999, 1.00184184, 1.002222679,
    1.004148161, 1.003218528, 1.009195353, 1.001426744,
    1.003984381, 1.025817745;

  for (size_t i = 0; i < 10; ++i) {
    auto rhats = chain_1.split_rank_normalized_rhat(i+7);
    EXPECT_NEAR(rhats.first, rhat_8_schools_1_bulk(i), 0.05);
    EXPECT_NEAR(rhats.second, rhat_8_schools_1_tail(i), 0.05);
  }
}

TEST_F(McmcChains, split_rank_normalized_ess) {
  std::vector<stan::io::stan_csv> eight_schools;
  eight_schools.push_back(eight_schools_1);
  eight_schools.push_back(eight_schools_2);
  stan::mcmc::chainset<> chain_2(eight_schools);
  EXPECT_EQ(2, chain_2.num_chains());

  // test against R implementation in pkg posterior
  Eigen::VectorXd ess_8_schools_bulk(10);
  ess_8_schools_bulk <<
    354, 287, 540, 673, 724, 174, 604, 355, 680, 71;
  Eigen::VectorXd ess_8_schools_tail(10);
  ess_8_schools_tail <<
    733, 395, 640, 386, 845, 564, 742, 646, 563, 71;

  for (size_t i = 0; i < 10; ++i) {
    auto ess = chain_2.split_rank_normalized_ess(i+7);
    EXPECT_NEAR(ess.first, std::round(ess_8_schools_bulk(i)), 5);
    EXPECT_NEAR(ess.second, ess_8_schools_tail(i), 5);
  }
}
