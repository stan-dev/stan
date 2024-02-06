#include <stan/mcmc/chains.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

class ComputeRhat : public testing::Test {
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

TEST_F(ComputeRhat, compute_potential_scale_reduction) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd rhat(48);
  rhat << 1.00067, 1.00497, 1.00918, 1.00055, 1.0015, 1.00088, 1.00776, 1.00042,
      1.00201, 0.999558, 0.99984, 1.00054, 1.00403, 1.00516, 1.00591, 1.00627,
      1.00134, 1.00895, 1.00079, 1.00368, 1.00092, 1.00133, 1.01005, 1.00107,
      1.00151, 1.00229, 0.999998, 1.00008, 1.00315, 1.00277, 1.00247, 1.00003,
      1.001, 1.01267, 1.00011, 1.00066, 1.00091, 1.00237, 1.00019, 1.00104,
      1.00341, 0.999815, 1.00033, 0.999672, 1.00306, 1.00072, 1.00191, 1.00658;

  // replicates calls to stan::analyze::compute_effective_sample_size
  // for any interface *without* access to chains class
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
    ASSERT_NEAR(rhat(index - 4),
                stan::analyze::compute_potential_scale_reduction(draws, sizes),
                1e-4)
        << "rhat for index: " << index
        << ", parameter: " << chains.param_name(index);
  }
}

TEST_F(ComputeRhat, compute_potential_scale_reduction_convenience) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd rhat(48);
  rhat << 1.00067, 1.00497, 1.00918, 1.00055, 1.0015, 1.00088, 1.00776, 1.00042,
      1.00201, 0.999558, 0.99984, 1.00054, 1.00403, 1.00516, 1.00591, 1.00627,
      1.00134, 1.00895, 1.00079, 1.00368, 1.00092, 1.00133, 1.01005, 1.00107,
      1.00151, 1.00229, 0.999998, 1.00008, 1.00315, 1.00277, 1.00247, 1.00003,
      1.001, 1.01267, 1.00011, 1.00066, 1.00091, 1.00237, 1.00019, 1.00104,
      1.00341, 0.999815, 1.00033, 0.999672, 1.00306, 1.00072, 1.00191, 1.00658;

  Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> samples(
      chains.num_chains());
  std::vector<const double*> draws(chains.num_chains());
  std::vector<size_t> sizes(chains.num_chains());
  for (int index = 4; index < chains.num_params(); index++) {
    for (int chain = 0; chain < chains.num_chains(); ++chain) {
      samples(chain) = chains.samples(chain, index);
      draws[chain] = &samples(chain)(0);
    }
    size_t size = samples(0).size();
    ASSERT_NEAR(rhat(index - 4),
                stan::analyze::compute_potential_scale_reduction(draws, size),
                1e-4)
        << "rhat for index: " << index
        << ", parameter: " << chains.param_name(index);
  }
}

TEST_F(ComputeRhat, chains_compute_split_potential_scale_reduction) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd rhat(48);
  rhat << 1.0078, 1.0109, 1.00731, 1.00333, 1.00401, 1.00992, 1.00734, 1.00633,
      1.00095, 1.00906, 1.01019, 1.00075, 1.00595, 1.00473, 1.00895, 1.01304,
      1.00166, 1.0074, 1.00236, 1.00588, 1.00414, 1.00303, 1.00976, 1.00295,
      1.00193, 1.0044, 1.00488, 1.00178, 1.01082, 1.0019, 1.00413, 1.01303,
      1.0024, 1.01148, 1.00436, 1.00515, 1.00712, 1.0089, 1.00222, 1.00118,
      1.00381, 1.00283, 1.00188, 1.00225, 1.00335, 1.00133, 1.00209, 1.0109;

  for (int index = 4; index < chains.num_params(); index++) {
    ASSERT_NEAR(rhat(index - 4), chains.split_potential_scale_reduction(index),
                1e-4)
        << "rhat for index: " << index
        << ", parameter: " << chains.param_name(index);
  }

  for (int index = 0; index < chains.num_params(); index++) {
    std::string name = chains.param_name(index);
    ASSERT_EQ(chains.split_potential_scale_reduction(index),
              chains.split_potential_scale_reduction(name));
  }
}

TEST_F(ComputeRhat, compute_split_potential_scale_reduction) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);

  chains.add(blocker2);

  Eigen::VectorXd rhat(48);
  rhat << 1.0078, 1.0109, 1.00731, 1.00333, 1.00401, 1.00992, 1.00734, 1.00633,
      1.00095, 1.00906, 1.01019, 1.00075, 1.00595, 1.00473, 1.00895, 1.01304,
      1.00166, 1.0074, 1.00236, 1.00588, 1.00414, 1.00303, 1.00976, 1.00295,
      1.00193, 1.0044, 1.00488, 1.00178, 1.01082, 1.0019, 1.00413, 1.01303,
      1.0024, 1.01148, 1.00436, 1.00515, 1.00712, 1.0089, 1.00222, 1.00118,
      1.00381, 1.00283, 1.00188, 1.00225, 1.00335, 1.00133, 1.00209, 1.0109;

  // replicates calls to stan::analyze::compute_effective_sample_size
  // for any interface *without* access to chains class
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
    ASSERT_NEAR(
        rhat(index - 4),
        stan::analyze::compute_split_potential_scale_reduction(draws, sizes),
        1e-4)
        << "rhat for index: " << index
        << ", parameter: " << chains.param_name(index);
  }
}

TEST_F(ComputeRhat, compute_split_potential_scale_reduction_convenience) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd rhat(48);
  rhat << 1.0078, 1.0109, 1.00731, 1.00333, 1.00401, 1.00992, 1.00734, 1.00633,
      1.00095, 1.00906, 1.01019, 1.00075, 1.00595, 1.00473, 1.00895, 1.01304,
      1.00166, 1.0074, 1.00236, 1.00588, 1.00414, 1.00303, 1.00976, 1.00295,
      1.00193, 1.0044, 1.00488, 1.00178, 1.01082, 1.0019, 1.00413, 1.01303,
      1.0024, 1.01148, 1.00436, 1.00515, 1.00712, 1.0089, 1.00222, 1.00118,
      1.00381, 1.00283, 1.00188, 1.00225, 1.00335, 1.00133, 1.00209, 1.0109;

  Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> samples(
      chains.num_chains());
  std::vector<const double*> draws(chains.num_chains());
  std::vector<size_t> sizes(chains.num_chains());
  for (int index = 4; index < chains.num_params(); index++) {
    for (int chain = 0; chain < chains.num_chains(); ++chain) {
      samples(chain) = chains.samples(chain, index);
      draws[chain] = &samples(chain)(0);
    }
    size_t size = samples(0).size();
    ASSERT_NEAR(
        rhat(index - 4),
        stan::analyze::compute_split_potential_scale_reduction(draws, size),
        1e-4)
        << "rhat for index: " << index
        << ", parameter: " << chains.param_name(index);
  }
}

TEST_F(ComputeRhat, compute_potential_scale_reduction_constant) {
  std::vector<std::string> param_names{"a"};
  stan::mcmc::chains<> chains(param_names);
  Eigen::Matrix<double, 2, 1> draws;
  draws << 1.0, 1.0;
  chains.add(draws);

  ASSERT_TRUE(std::isnan(chains.split_potential_scale_reduction(0)))
      << "rhat for index: " << 1 << ", parameter: " << chains.param_name(1);
}

TEST_F(ComputeRhat, compute_potential_scale_reduction_nan) {
  std::vector<std::string> param_names{"a"};
  stan::mcmc::chains<> chains(param_names);
  Eigen::Matrix<double, 2, 1> draws;
  draws << 1.0, std::numeric_limits<double>::quiet_NaN();
  chains.add(draws);

  ASSERT_TRUE(std::isnan(chains.split_potential_scale_reduction(0)))
      << "rhat for index: " << 1 << ", parameter: " << chains.param_name(1);
}
