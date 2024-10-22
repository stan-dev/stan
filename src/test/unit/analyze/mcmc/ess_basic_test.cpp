#include <stan/analyze/mcmc/ess.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

class EssBasic : public testing::Test {
 public:
  void SetUp() {
    chains_lp.resize(1000, 4);
    chains_theta.resize(1000, 4);
    draws_theta.resize(4);
    draws_lp.resize(4);
    sizes.resize(4);
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
      draws_lp[i] = chains_lp.col(i).data();
      draws_theta[i] = chains_theta.col(i).data();
      sizes[i] = 1000;
    }
  }

  void TearDown() {}

  std::stringstream out;
  Eigen::MatrixXd chains_lp;
  Eigen::MatrixXd chains_theta;
  std::vector<const double*> draws_theta;
  std::vector<const double*> draws_lp;
  std::vector<size_t> sizes;
};

TEST_F(EssBasic, test_basic_ess) {
  // computed via cmdstan 2.35.0 stansummary
  double ess_lp_expect = 1335.41366787;
  double ess_theta_expect = 1377.5030282;

  auto ess_basic_lp = stan::analyze::ess(chains_lp);
  auto ess_basic_theta = stan::analyze::ess(chains_theta);

  EXPECT_NEAR(ess_lp_expect, ess_basic_lp, 1e-8);
  EXPECT_NEAR(ess_theta_expect, ess_basic_theta, 1e-8);
}
