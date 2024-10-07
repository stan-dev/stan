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
  rhat << 1.00042, 1.00036, 0.99955, 1.00047, 1.00119, 1.00089, 1.00018,
      1.00019, 1.00226, 0.99954, 0.9996, 0.99951, 1.00237, 1.00515, 1.00566,
      0.99957, 1.00099, 1.00853, 1.0008, 0.99961, 1.0006, 1.00046, 1.01023,
      0.9996, 1.0011, 0.99967, 0.99973, 0.99958, 1.00242, 1.00213, 1.00244,
      0.99998, 0.99969, 1.00079, 0.99955, 1.0009, 1.00136, 1.00288, 1.00036,
      0.99989, 1.00077, 0.99997, 1.00194, 0.99972, 1.00257, 1.00109, 1.00004,
      0.99955;

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
  rhat << 1.00042, 1.00036, 0.99955, 1.00047, 1.00119, 1.00089, 1.00018,
      1.00019, 1.00226, 0.99954, 0.9996, 0.99951, 1.00237, 1.00515, 1.00566,
      0.99957, 1.00099, 1.00853, 1.0008, 0.99961, 1.0006, 1.00046, 1.01023,
      0.9996, 1.0011, 0.99967, 0.99973, 0.99958, 1.00242, 1.00213, 1.00244,
      0.99998, 0.99969, 1.00079, 0.99955, 1.0009, 1.00136, 1.00288, 1.00036,
      0.99989, 1.00077, 0.99997, 1.00194, 0.99972, 1.00257, 1.00109, 1.00004,
      0.99955;

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
  rhat << 1.00718, 1.00473, 0.999203, 1.00061, 1.00378, 1.01031, 1.00173,
      1.0045, 1.00111, 1.00337, 1.00546, 1.00105, 1.00558, 1.00463, 1.00534,
      1.01244, 1.00174, 1.00718, 1.00186, 1.00554, 1.00436, 1.00147, 1.01017,
      1.00162, 1.00143, 1.00058, 0.999221, 1.00012, 1.01028, 1.001, 1.00305,
      1.00435, 1.00055, 1.00246, 1.00447, 1.0048, 1.00209, 1.01159, 1.00202,
      1.00077, 1.0021, 1.00262, 1.00308, 1.00197, 1.00246, 1.00085, 1.00047,
      1.00735;

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
  rhat << 1.00718, 1.00473, 0.999203, 1.00061, 1.00378, 1.01031, 1.00173,
      1.0045, 1.00111, 1.00337, 1.00546, 1.00105, 1.00558, 1.00463, 1.00534,
      1.01244, 1.00174, 1.00718, 1.00186, 1.00554, 1.00436, 1.00147, 1.01017,
      1.00162, 1.00143, 1.00058, 0.999221, 1.00012, 1.01028, 1.001, 1.00305,
      1.00435, 1.00055, 1.00246, 1.00447, 1.0048, 1.00209, 1.01159, 1.00202,
      1.00077, 1.0021, 1.00262, 1.00308, 1.00197, 1.00246, 1.00085, 1.00047,
      1.00735;

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
  rhat << 1.00718, 1.00473, 0.999203, 1.00061, 1.00378, 1.01031, 1.00173,
      1.0045, 1.00111, 1.00337, 1.00546, 1.00105, 1.00558, 1.00463, 1.00534,
      1.01244, 1.00174, 1.00718, 1.00186, 1.00554, 1.00436, 1.00147, 1.01017,
      1.00162, 1.00143, 1.00058, 0.999221, 1.00012, 1.01028, 1.001, 1.00305,
      1.00435, 1.00055, 1.00246, 1.00447, 1.0048, 1.00209, 1.01159, 1.00202,
      1.00077, 1.0021, 1.00262, 1.00308, 1.00197, 1.00246, 1.00085, 1.00047,
      1.00735;

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
