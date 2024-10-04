#include <stan/mcmc/chains.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

class RankNormalizedRhat : public testing::Test {
 public:
  void SetUp() {
    eight_schools_1_stream.open(
        "src/test/unit/mcmc/test_csv_files/eight_schools_1.csv",
        std::ifstream::in);
    eight_schools_2_stream.open(
        "src/test/unit/mcmc/test_csv_files/eight_schools_2.csv",
        std::ifstream::in);
    if (!eight_schools_1_stream || !eight_schools_2_stream) {
      FAIL() << "Failed to open one or more test files";
    }
    eight_schools_1_stream.seekg(0, std::ios::beg);
    eight_schools_2_stream.seekg(0, std::ios::beg);
    eight_schools_1
        = stan::io::stan_csv_reader::parse(eight_schools_1_stream, &out);
    eight_schools_2
        = stan::io::stan_csv_reader::parse(eight_schools_2_stream, &out);
  }

  void TearDown() {
    blocker1_stream.close();
    blocker2_stream.close();
  }

  std::stringstream out;
  std::ifstream eight_schools_1_stream, eight_schools_2_stream;
  stan::io::stan_csv eight_schools_1, eight_schools_2;
};


TEST_F(RankNormalizedRhat, compute_split_rank_normalized_rhat) {
  stan::mcmc::chains<> chains(eight_schools_1);

  // test against R implementation in pkg posterior
  Eigen::VectorXd rhat_8_schools_1_bulk(10);
  rhat_8_schools_1_bulk << 1.0012958313, 1.0046136496, 1.0085723580,
      1.0248629375, 1.0111456620, 1.0004458336, 0.9987162973, 1.0339773469,
      0.9985612618, 1.0281667351;

  Eigen::VectorXd rhat_8_schools_1_tail(10);
  rhat_8_schools_1_tail << 1.005676523, 1.009670999, 1.00184184, 1.002222679,
      1.004148161, 1.003218528, 1.009195353, 1.001426744, 1.003984381,
      1.025817745;

  for (size_t i = 0; i < 10; ++i) {
    auto rhats = chains.split_rank_normalized_rhat(i + 7);
    EXPECT_NEAR(rhats.first, rhat_8_schools_1_bulk(i), 0.05);
    EXPECT_NEAR(rhats.second, rhat_8_schools_1_tail(i), 0.05);
  }
}

TEST_F(RankNormalizedRhat, const_fail) {
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
  stan::mcmc::chains<> chains(bernoulli_const_1);
  chains.add(bernoulli_const_2);
  
  auto rhat = chains.split_rank_normalized_rhat("zeta");
  EXPECT_TRUE(std::isnan(rhat.first));
  EXPECT_TRUE(std::isnan(rhat.second));
}
