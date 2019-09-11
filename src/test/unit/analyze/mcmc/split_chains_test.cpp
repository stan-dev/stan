#include <stan/mcmc/chains.hpp>
#include <stan/analyze/mcmc/split_chains.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>

class SplitChains : public testing::Test {
 public:
  void SetUp() {
    blocker1_stream.open("src/test/unit/mcmc/test_csv_files/blocker.1.csv");
    blocker2_stream.open("src/test/unit/mcmc/test_csv_files/blocker.2.csv");
  }

  void TearDown() {
    blocker1_stream.close();
    blocker2_stream.close();
  }
  std::ifstream blocker1_stream, blocker2_stream;
};

TEST_F(SplitChains, split_chains) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> samples(
      chains.num_chains());
  std::vector<const double*> draws(chains.num_chains());
  std::vector<size_t> sizes(chains.num_chains());
  for (int index = 4; index < chains.num_params(); index++) {
    for (int chain = 0; chain < chains.num_chains(); ++chain) {
      samples(chain) = chains.samples(chain, index);
      draws[chain] = &samples(chain)(0);
      sizes[chain] = samples(chain).size();
    }
  }

  std::vector<const double*> split_draws
      = stan::analyze::split_chains(draws, sizes);
  for (int chain = 0; chain < chains.num_chains(); ++chain) {
    double half = sizes[chain] / 2.0;
    int first_half = std::floor(half);
    for (int draw = 0; draw < first_half; ++draw) {
      ASSERT_NEAR(samples(chain)(draw), split_draws[chain][draw], 1.0)
          << "samples[" << chain << "][" << draw << "]: " << draws[chain][draw]
          << ", split_chain[" << chain << "][" << draw
          << "]: " << split_draws[chain][draw];
    }

    int second_half = std::ceil(half);
    for (int draw = second_half; draw < sizes[chain]; ++draw) {
      ASSERT_NEAR(samples(chain)(draw),
                  split_draws[2 * chain + 1][draw - first_half], 1.0)
          << "samples[" << chain << "][" << draw << "]: " << draws[chain][draw]
          << ", split_chain[" << chain << "][" << draw
          << "]: " << split_draws[chain][draw];
    }
  }
}

TEST_F(SplitChains, split_chains_convenience) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> samples(
      chains.num_chains());
  std::vector<const double*> draws(chains.num_chains());
  for (int index = 4; index < chains.num_params(); index++) {
    for (int chain = 0; chain < chains.num_chains(); ++chain) {
      samples(chain) = chains.samples(chain, index);
      draws[chain] = &samples(chain)(0);
    }
  }
  size_t size = samples(0).size();

  std::vector<const double*> split_draws
      = stan::analyze::split_chains(draws, size);
  for (int chain = 0; chain < chains.num_chains(); ++chain) {
    double half = size / 2.0;
    int first_half = std::floor(half);
    for (int draw = 0; draw < first_half; ++draw) {
      ASSERT_NEAR(samples(chain)(draw), split_draws[chain][draw], 1.0)
          << "samples[" << chain << "][" << draw << "]: " << draws[chain][draw]
          << ", split_chain[" << chain << "][" << draw
          << "]: " << split_draws[chain][draw];
    }

    int second_half = std::ceil(half);
    for (int draw = second_half; draw < size; ++draw) {
      ASSERT_NEAR(samples(chain)(draw),
                  split_draws[2 * chain + 1][draw - first_half], 1.0)
          << "samples[" << chain << "][" << draw << "]: " << draws[chain][draw]
          << ", split_chain[" << chain << "][" << draw
          << "]: " << split_draws[chain][draw];
    }
  }
}
