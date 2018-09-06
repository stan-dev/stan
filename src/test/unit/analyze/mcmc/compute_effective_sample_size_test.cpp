#include <stan/mcmc/chains.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>

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

TEST_F(ComputeEss,compute_effective_sample_size) {
  std::stringstream out;
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2 = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd n_eff(48);
  n_eff << 466.099,136.953,1170.390,541.256,
    518.051,589.244,764.813,688.294,
    323.777,502.892,353.823,588.142,
    654.336,480.914,176.978,182.649,
    642.389,470.949,561.947,581.187,
    446.389,397.641,338.511,678.772,
    1442.250,837.956,869.865,951.124,
    619.336,875.805,233.260,786.568,
    910.144,231.582,907.666,747.347,
    720.660,195.195,944.547,767.271,
    723.665,1077.030,470.903,954.924,
    497.338,583.539,697.204,98.421;

  // replicates calls to stan::anlayze::compute_effective_sample_size
  // for any interface *without* access to chains class
  Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1>
    samples(chains.num_chains());
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
      << "n_effective for index: " << index << ", parameter: "
      << chains.param_name(index);
  }
}

TEST_F(ComputeEss,compute_effective_sample_size_convenience) {
  std::stringstream out;
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2 = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd n_eff(48);
  n_eff << 466.099,136.953,1170.390,541.256,
    518.051,589.244,764.813,688.294,
    323.777,502.892,353.823,588.142,
    654.336,480.914,176.978,182.649,
    642.389,470.949,561.947,581.187,
    446.389,397.641,338.511,678.772,
    1442.250,837.956,869.865,951.124,
    619.336,875.805,233.260,786.568,
    910.144,231.582,907.666,747.347,
    720.660,195.195,944.547,767.271,
    723.665,1077.030,470.903,954.924,
    497.338,583.539,697.204,98.421;

  Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1>
    samples(chains.num_chains());
  std::vector<const double*> draws(chains.num_chains());
  std::vector<size_t> sizes(chains.num_chains());
  for (int index = 4; index < chains.num_params(); index++) {
    for (int chain = 0; chain < chains.num_chains(); ++chain) {
      samples(chain) = chains.samples(chain, index);
      draws[chain] = &samples(chain)(0);

    }
    size_t size  = samples(0).size();
    ASSERT_NEAR(n_eff(index - 4),
                stan::analyze::compute_effective_sample_size(draws, size), 1.0)
      << "n_effective for index: " << index << ", parameter: "
      << chains.param_name(index);
  }
}

TEST_F(ComputeEss,chains_compute_effective_sample_size) {
  std::stringstream out;
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2 = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd n_eff(48);
  n_eff << 466.099,136.953,1170.390,541.256,
    518.051,589.244,764.813,688.294,
    323.777,502.892,353.823,588.142,
    654.336,480.914,176.978,182.649,
    642.389,470.949,561.947,581.187,
    446.389,397.641,338.511,678.772,
    1442.250,837.956,869.865,951.124,
    619.336,875.805,233.260,786.568,
    910.144,231.582,907.666,747.347,
    720.660,195.195,944.547,767.271,
    723.665,1077.030,470.903,954.924,
    497.338,583.539,697.204,98.421;

  // replicates calls to stan::anlayze::compute_effective_sample_size
  // for any interface with access to chains class
  for (int index = 4; index < chains.num_params(); index++) {
    ASSERT_NEAR(n_eff(index - 4),
                chains.effective_sample_size(index), 1.0)
      << "n_effective for index: " << index << ", parameter: "
      << chains.param_name(index);
  }

  for (int index = 0; index < chains.num_params(); index++) {
    std::string name = chains.param_name(index);
    ASSERT_EQ(chains.effective_sample_size(index),
              chains.effective_sample_size(name));
  }
}
