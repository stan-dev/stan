#include <stan/analyze/mcmc/rank_normalization.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <limits>

class RankNormalization : public testing::Test {
 public:
  void SetUp() {
    chains_theta.resize(1000, 4);
    for (size_t i = 0; i < 4; ++i) {
      std::stringstream fname;
      fname << "src/test/unit/analyze/mcmc/test_csv_files/bern" << (i + 1)
            << ".csv";
      std::ifstream bern_stream(fname.str(), std::ifstream::in);
      stan::io::stan_csv bern_csv
          = stan::io::stan_csv_reader::parse(bern_stream, &out);
      bern_stream.close();
      chains_theta.col(i) = bern_csv.samples.col(7);
    }
  }

  void TearDown() {}

  std::stringstream out;
  Eigen::MatrixXd chains_theta;
};

TEST_F(RankNormalization, test_min_max) {
  Eigen::Index maxRow, maxCol;
  Eigen::Index minRow, minCol;
  double max_val = chains_theta.maxCoeff(&maxRow, &maxCol);
  double min_val = chains_theta.minCoeff(&minRow, &minCol);
  chains_theta(maxRow, maxCol) = 0.9999;
  chains_theta(minRow, minCol) = 0.001;

  auto rank_norm_theta = stan::analyze::rank_transform(chains_theta);

  Eigen::Index maxRowRankNorm, maxColRankNorm;
  Eigen::Index minRowRankNorm, minColRankNorm;
  max_val = rank_norm_theta.maxCoeff(&maxRowRankNorm, &maxColRankNorm);
  min_val = rank_norm_theta.minCoeff(&minRowRankNorm, &minColRankNorm);
  EXPECT_EQ(maxRow, maxRowRankNorm);
  EXPECT_EQ(minRow, minRowRankNorm);
  EXPECT_EQ(maxCol, maxColRankNorm);
  EXPECT_EQ(minCol, minColRankNorm);
}

TEST_F(RankNormalization, test_symmetry) {
  Eigen::MatrixXd foo(8, 2);
  int val = 0;
  for (size_t col = 0; col < 2; ++col) {
    for (size_t row = 0; row < 8; ++row) {
      val++;
      foo(row, col) = val;
    }
  }
  auto bar = stan::analyze::rank_transform(foo);
  size_t rev_col = 2;
  for (size_t col = 0; col < 2; ++col) {
    --rev_col;
    size_t rev_row = 8;
    for (size_t row = 0; row < 8; ++row) {
      --rev_row;
      EXPECT_NEAR(bar(row, col), -bar(rev_row, rev_col), 1e-10);
    }
  }

  Eigen::MatrixXd baz(2, 8);
  for (size_t col = 0; col < 8; ++col) {
    int val = 0;
    for (size_t row = 0; row < 2; ++row) {
      val++;
      baz(row, col) = val;
    }
  }
  auto boz = stan::analyze::rank_transform(baz);
  for (size_t col = 0; col < 8; ++col) {
    EXPECT_NEAR(boz(0, col), -boz(1, col), 1e-10);
  }
}
