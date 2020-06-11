#include <stan/mcmc/chains.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <cmath>

class ComputeEss : public testing::Test {
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

TEST_F(ComputeEss, compute_effective_sample_size_one_chain) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);

  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);

  Eigen::VectorXd n_eff(49);
  n_eff << 284.772, 105.681, 668.691, 569.402, 523.292, 403.396, 432.348,
      441.288, 209.865, 472.828, 451.133, 429.327, 375.418, 507.376, 222.906,
      218.278, 316.072, 489.084, 404.057, 379.351, 232.849, 445.684, 675.562,
      362.882, 720.201, 426.744, 376.692, 509.399, 247.151, 440.426, 160.532,
      411.109, 419.395, 411.988, 425.529, 420.615, 336.495, 131.946, 461.606,
      469.628, 479.458, 611.196, 483.301, 584.614, 500.264, 453.112, 646.067,
      72.1878, 1;

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
    ASSERT_NEAR(n_eff(index - 4),
                stan::analyze::compute_effective_sample_size(draws, sizes), 1.0)
        << "n_effective for index: " << index
        << ", parameter: " << chains.param_name(index);
  }
}

TEST_F(ComputeEss, compute_effective_sample_size_two_chains) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd n_eff(49);
  n_eff << 467.368, 138.628, 1171.629, 543.893, 519.897, 590.533, 764.757,
      690.219, 326.217, 505.51, 356.445, 590.149, 655.714, 480.728, 178.746,
      184.871, 643.856, 472.13, 563.848, 584.745, 449.137, 400.235, 339.217,
      680.605, 1410.383, 836.017, 871.39, 952.265, 620.944, 869.979, 235.168,
      788.52, 911.348, 234.228, 909.209, 748.71, 722.362, 196.762, 945.741,
      768.797, 725.527, 1078.467, 471.57, 956.357, 498.195, 582.663, 696.851,
      99.784, 80.52;

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
    ASSERT_NEAR(n_eff(index - 4),
                stan::analyze::compute_effective_sample_size(draws, sizes), 1.0)
        << "n_effective for index: " << index
        << ", parameter: " << chains.param_name(index);
  }
}

TEST_F(ComputeEss, compute_effective_sample_size_convenience) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd n_eff(49);
  n_eff << 467.368, 138.628, 1171.629, 543.893, 519.897, 590.533, 764.757,
      690.219, 326.217, 505.51, 356.445, 590.149, 655.714, 480.728, 178.746,
      184.871, 643.856, 472.13, 563.848, 584.745, 449.137, 400.235, 339.217,
      680.605, 1410.383, 836.017, 871.39, 952.265, 620.944, 869.979, 235.168,
      788.52, 911.348, 234.228, 909.209, 748.71, 722.362, 196.762, 945.741,
      768.797, 725.527, 1078.467, 471.57, 956.357, 498.195, 582.663, 696.851,
      99.784, 80.52;

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
    ASSERT_NEAR(n_eff(index - 4),
                stan::analyze::compute_effective_sample_size(draws, size), 1.0)
        << "n_effective for index: " << index
        << ", parameter: " << chains.param_name(index);
  }
}

TEST_F(ComputeEss, chains_compute_effective_sample_size) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd n_eff(49);
  n_eff << 467.368, 138.628, 1171.629, 543.893, 519.897, 590.533, 764.757,
      690.219, 326.217, 505.51, 356.445, 590.149, 655.714, 480.728, 178.746,
      184.871, 643.856, 472.13, 563.848, 584.745, 449.137, 400.235, 339.217,
      680.605, 1410.383, 836.017, 871.39, 952.265, 620.944, 869.979, 235.168,
      788.52, 911.348, 234.228, 909.209, 748.71, 722.362, 196.762, 945.741,
      768.797, 725.527, 1078.467, 471.57, 956.357, 498.195, 582.663, 696.851,
      99.784, 80.52;

  // replicates calls to stan::analyze::compute_effective_sample_size
  // for any interface with access to chains class
  for (int index = 4; index < chains.num_params(); index++) {
    ASSERT_NEAR(n_eff(index - 4), chains.effective_sample_size(index), 1.0)
        << "n_effective for index: " << index
        << ", parameter: " << chains.param_name(index);
  }

  for (int index = 0; index < chains.num_params(); index++) {
    std::string name = chains.param_name(index);
    ASSERT_EQ(chains.effective_sample_size(index),
              chains.effective_sample_size(name));
  }
}

TEST_F(ComputeEss, compute_split_effective_sample_size) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd n_eff(49);
  n_eff << 467.845, 134.498, 1189.591, 569.193, 525.002, 572.692, 763.91,
      710.977, 338.298, 493.348, 333.493, 588.283, 665.62, 504.263, 187.049,
      156.913, 650.018, 501.455, 570.161, 550.366, 446.219, 408.218, 364.204,
      678.699, 1419.234, 841.742, 881.923, 960.42, 610.921, 917.642, 239.599,
      773.726, 921.332, 227.34, 900.819, 748.478, 727.365, 184.949, 948.425,
      776.03, 735.279, 1077.177, 475.252, 955.281, 503.045, 591.913, 715.97,
      95.594, 77.904;

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
        n_eff(index - 4),
        stan::analyze::compute_split_effective_sample_size(draws, sizes), 1.0)
        << "n_effective for index: " << index
        << ", parameter: " << chains.param_name(index);
  }
}

TEST_F(ComputeEss, compute_split_effective_sample_size_convenience) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd n_eff(49);
  n_eff << 467.845, 134.498, 1189.591, 569.193, 525.002, 572.692, 763.91,
      710.977, 338.298, 493.348, 333.493, 588.283, 665.62, 504.263, 187.049,
      156.913, 650.018, 501.455, 570.161, 550.366, 446.219, 408.218, 364.204,
      678.699, 1419.234, 841.742, 881.923, 960.42, 610.921, 917.642, 239.599,
      773.726, 921.332, 227.34, 900.819, 748.478, 727.365, 184.949, 948.425,
      776.03, 735.279, 1077.177, 475.252, 955.281, 503.045, 591.913, 715.97,
      95.594, 77.904;

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
    ASSERT_NEAR(n_eff(index - 4),
                stan::analyze::compute_split_effective_sample_size(draws, size),
                1.0)
        << "n_effective for index: " << index
        << ", parameter: " << chains.param_name(index);
  }
}

TEST_F(ComputeEss, chains_compute_split_effective_sample_size) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd n_eff(49);
  n_eff << 467.845, 134.498, 1189.591, 569.193, 525.002, 572.692, 763.91,
      710.977, 338.298, 493.348, 333.493, 588.283, 665.62, 504.263, 187.049,
      156.913, 650.018, 501.455, 570.161, 550.366, 446.219, 408.218, 364.204,
      678.699, 1419.234, 841.742, 881.923, 960.42, 610.921, 917.642, 239.599,
      773.726, 921.332, 227.34, 900.819, 748.478, 727.365, 184.949, 948.425,
      776.03, 735.279, 1077.177, 475.252, 955.281, 503.045, 591.913, 715.97,
      95.594, 77.904;

  // replicates calls to stan::analyze::compute_effective_sample_size
  // for any interface with access to chains class
  for (int index = 4; index < chains.num_params(); index++) {
    ASSERT_NEAR(n_eff(index - 4), chains.split_effective_sample_size(index),
                1.0)
        << "n_effective for index: " << index
        << ", parameter: " << chains.param_name(index);
  }

  for (int index = 0; index < chains.num_params(); index++) {
    std::string name = chains.param_name(index);
    ASSERT_EQ(chains.split_effective_sample_size(index),
              chains.split_effective_sample_size(name));
  }
}

TEST_F(ComputeEss, compute_effective_sample_size_minimum_n) {
  std::vector<std::string> param_names{"a"};
  stan::mcmc::chains<> chains(param_names);
  Eigen::Matrix<double, 3, 1> draws;
  draws << 1.0, 2.0, 3.0;
  chains.add(draws);

  ASSERT_TRUE(std::isnan(chains.effective_sample_size(0)))
      << "n_effective for index: " << 0
      << ", parameter: " << chains.param_name(0);
}

TEST_F(ComputeEss, compute_effective_sample_size_sufficient_n) {
  std::vector<std::string> param_names{"a"};
  stan::mcmc::chains<> chains(param_names);
  Eigen::Matrix<double, 4, 1> draws;
  draws << 1.0, 2.0, 3.0, 4.0;
  chains.add(draws);

  ASSERT_TRUE(!std::isnan(chains.effective_sample_size(0)))
      << "n_effective for index: " << 0
      << ", parameter: " << chains.param_name(0);
}

TEST_F(ComputeEss, compute_effective_sample_size_nan) {
  std::vector<std::string> param_names{"a"};
  stan::mcmc::chains<> chains(param_names);
  Eigen::Matrix<double, 2, 1> draws;
  draws << 1.0, std::numeric_limits<double>::quiet_NaN();
  chains.add(draws);

  ASSERT_TRUE(std::isnan(chains.effective_sample_size(0)))
      << "n_effective for index: " << 0
      << ", parameter: " << chains.param_name(0);
}

TEST_F(ComputeEss, compute_effective_sample_size_constant) {
  std::vector<std::string> param_names{"a"};
  stan::mcmc::chains<> const_chains(param_names);
  Eigen::Matrix<double, 4, 1> const_draws;
  const_draws << 1.0, 1.0, 1.0, 1.0;
  const_chains.add(const_draws);

  ASSERT_TRUE(std::isnan(const_chains.effective_sample_size(0)))
      << "n_effective for index: " << 0
      << ", parameter: " << const_chains.param_name(0);
}

TEST_F(ComputeEss, compute_effective_sample_size_not_constant) {
  std::vector<std::string> param_names{"a"};
  stan::mcmc::chains<> nonconst_chains(param_names);
  Eigen::Matrix<double, 4, 1> nonconst_draws;
  nonconst_draws << 1.0, 2.0, 3.0, 4.0;
  nonconst_chains.add(nonconst_draws);

  ASSERT_TRUE(!std::isnan(nonconst_chains.effective_sample_size(0)))
      << "n_effective for index: " << 0
      << ", parameter: " << nonconst_chains.param_name(0);
}
