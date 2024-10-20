#include <stan/analyze/mcmc/check_chains.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

TEST(CheckChains, good_and_bad) {
  std::stringstream out;
  std::ifstream eight_schools_1_stream, eight_schools_2_stream;
  stan::io::stan_csv eight_schools_1, eight_schools_2;
  eight_schools_1_stream.open(
      "src/test/unit/mcmc/test_csv_files/eight_schools_1.csv",
      std::ifstream::in);
  eight_schools_1
      = stan::io::stan_csv_reader::parse(eight_schools_1_stream, &out);
  eight_schools_1_stream.close();

  eight_schools_2_stream.open(
      "src/test/unit/mcmc/test_csv_files/eight_schools_2.csv",
      std::ifstream::in);
  eight_schools_2
      = stan::io::stan_csv_reader::parse(eight_schools_2_stream, &out);
  eight_schools_2_stream.close();

  Eigen::MatrixXd chain_1(eight_schools_1.samples.rows(), 1);
  Eigen::MatrixXd chains(eight_schools_1.samples.rows(), 2);

  // stepsize - constant after adaptation
  chain_1.col(0) = eight_schools_1.samples.col(2);
  EXPECT_FALSE(stan::analyze::is_finite_and_varies(chain_1));

  chains.col(0) = eight_schools_1.samples.col(2);
  chains.col(1) = eight_schools_2.samples.col(2);
  EXPECT_FALSE(stan::analyze::is_finite_and_varies(chains));

  for (size_t i = 0; i < 10; ++i) {
    chains.col(0) = eight_schools_1.samples.col(i + 7);
    chains.col(1) = eight_schools_2.samples.col(i + 7);
    EXPECT_TRUE(stan::analyze::is_finite_and_varies(chains));
  }

  // above test shows that column 7 is OK - make it non-finite
  chain_1.col(0) = eight_schools_1.samples.col(7);
  chain_1(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(stan::analyze::is_finite_and_varies(chain_1));
}
