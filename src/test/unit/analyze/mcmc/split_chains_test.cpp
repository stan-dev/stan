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

TEST_F(SplitChains, split_draws_matrix_odd_rows) {
  //  When the number of total draws N is odd, the (N+1)/2th draw is ignored.
  Eigen::MatrixXd foo(7, 2);
  int val = 0;
  for (size_t col = 0; col < 2; ++col) {
    for (size_t row = 0; row < 7; ++row) {
      val += 1;
      foo(row, col) = val;
    }
  }
  auto bar = stan::analyze::split_chains(foo);
  EXPECT_EQ(4, bar.cols());
  EXPECT_EQ(3, bar.rows());
  EXPECT_EQ(bar(0, 0), 1);
  EXPECT_EQ(bar(1, 0), 2);
  EXPECT_EQ(bar(2, 0), 3);
  EXPECT_EQ(bar(0, 1), 5);
  EXPECT_EQ(bar(1, 1), 6);
  EXPECT_EQ(bar(2, 1), 7);
  EXPECT_EQ(bar(0, 2), 8);
  EXPECT_EQ(bar(1, 2), 9);
  EXPECT_EQ(bar(2, 2), 10);
  EXPECT_EQ(bar(0, 3), 12);
  EXPECT_EQ(bar(1, 3), 13);
  EXPECT_EQ(bar(2, 3), 14);

  Eigen::MatrixXd baz(4, 2);
  for (size_t col = 0; col < 2; ++col) {
    for (size_t row = 0; row < 4; ++row) {
      val += 1;
      baz(row, col) = val;
    }
  }
  auto boz = stan::analyze::split_chains(baz);
  EXPECT_EQ(4, boz.cols());
  EXPECT_EQ(2, boz.rows());
  EXPECT_EQ(boz(0, 0), 15);
  EXPECT_EQ(boz(1, 0), 16);
  EXPECT_EQ(boz(0, 1), 17);
  EXPECT_EQ(boz(1, 1), 18);
  EXPECT_EQ(boz(0, 2), 19);
  EXPECT_EQ(boz(1, 2), 20);
  EXPECT_EQ(boz(0, 3), 21);
  EXPECT_EQ(boz(1, 3), 22);
}
