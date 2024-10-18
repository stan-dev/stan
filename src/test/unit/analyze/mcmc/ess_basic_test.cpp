#include <stan/analyze/mcmc/compute_effective_sample_size.hpp>
#include <stan/analyze/mcmc/ess.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

TEST(RankNormalizedEss, test_basic_ess) {
  std::stringstream out;
  Eigen::MatrixXd chains_lp(1000, 4);
  Eigen::MatrixXd chains_theta(1000, 4);

  std::vector<const double*> draws_theta(4);
  std::vector<const double*> draws_lp(4);
  std::vector<size_t> sizes(4);

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
  double ess_lp_expect = 1335.4137;
  double ess_theta_expect = 1377.503;

  auto ess_basic_lp = stan::analyze::ess(chains_lp);
  auto old_ess_basic_lp
      = stan::analyze::compute_effective_sample_size(draws_lp, sizes);

  auto ess_basic_theta = stan::analyze::ess(chains_theta);
  auto old_ess_basic_theta
      = stan::analyze::compute_effective_sample_size(draws_theta, sizes);

  EXPECT_NEAR(ess_lp_expect, ess_basic_lp, 0.0001);
  EXPECT_NEAR(ess_theta_expect, ess_basic_theta, 0.0001);

  EXPECT_NEAR(old_ess_basic_lp, ess_basic_lp, 0.00001);
  EXPECT_NEAR(old_ess_basic_theta, ess_basic_theta, 0.00001);
}
