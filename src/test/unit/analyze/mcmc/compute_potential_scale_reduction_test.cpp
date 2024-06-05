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
TEST_F(ComputeRhat, compute_potential_scale_reduction_rank) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  // Eigen::VectorXd rhat(48);
  // rhat
  // << 1.00067,1.00497,1.00918,1.00055,1.0015,1.00088,1.00776,1.00042,1.00201,0.99956,0.99984,1.00054,1.00403,1.00516,1.00591,1.00627,1.00134,1.00895,1.00079,1.00368,1.00092,1.00133,1.01005,1.00107,1.00151,1.00229,1.0,1.00008,1.00315,1.00277,1.00247,1.00003,1.001,1.01267,1.00011,1.00066,1.00091,1.00237,1.00019,1.00104,1.00341,0.99981,1.00033,0.99967,1.00306,1.00072,1.00191,1.00658;

  Eigen::VectorXd rhat_bulk(48);
  rhat_bulk << 1.00067, 0.99979, 0.99966, 1.00055, 1.0011, 1.00088, 1.00032,
      0.99997, 1.00201, 0.99956, 0.99956, 0.9995, 1.00292, 1.00516, 1.00591,
      0.99975, 1.00088, 1.00895, 1.00079, 0.99953, 1.00092, 1.00044, 1.01005,
      0.9996, 1.00151, 0.99966, 0.99965, 0.99963, 1.00315, 1.00277, 1.00247,
      1.00003, 0.99994, 1.00116, 0.99952, 1.0005, 1.00091, 1.00213, 1.00019,
      0.99977, 1.0003, 0.99981, 1.00003, 0.99967, 1.00306, 1.00072, 0.9996,
      0.99979;
  Eigen::VectorXd rhat_tail(48);
  rhat_tail << 1.00063, 1.00497, 1.00918, 0.99965, 1.0015, 0.99962, 1.00776,
      1.00042, 0.99963, 0.99951, 0.99984, 1.00054, 1.00403, 1.00107, 1.00287,
      1.00627, 1.00134, 0.99957, 0.99997, 1.00368, 1.00053, 1.00133, 1.00589,
      1.00107, 1.00031, 1.00229, 1.0, 1.00008, 1.0001, 1.00116, 1.00219,
      0.99992, 1.001, 1.01267, 1.00011, 1.00066, 1.00065, 1.00237, 0.9995,
      1.00104, 1.00341, 0.99958, 1.00033, 0.9996, 0.99957, 1.00058, 1.00191,
      1.00658;

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
    double computed_bulk_rhat, computed_tail_rhat;
    std::tie(computed_bulk_rhat, computed_tail_rhat)
        = stan::analyze::compute_potential_scale_reduction_rank(draws, sizes);
    double expected_bulk_rhat = rhat_bulk(index - 4);
    double expected_tail_rhat = rhat_tail(index - 4);

    ASSERT_NEAR(expected_bulk_rhat, computed_bulk_rhat, 1e-4)
        << "Bulk Rhat mismatch for index: " << index
        << ", parameter: " << chains.param_name(index);
    ASSERT_NEAR(expected_tail_rhat, computed_tail_rhat, 1e-4)
        << "Tail Rhat mismatch for index: " << index
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

TEST_F(ComputeRhat, compute_potential_scale_reduction_rank_convenience) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd rhat_bulk(48);
  rhat_bulk << 1.00067, 0.99979, 0.99966, 1.00055, 1.0011, 1.00088, 1.00032,
      0.99997, 1.00201, 0.99956, 0.99956, 0.9995, 1.00292, 1.00516, 1.00591,
      0.99975, 1.00088, 1.00895, 1.00079, 0.99953, 1.00092, 1.00044, 1.01005,
      0.9996, 1.00151, 0.99966, 0.99965, 0.99963, 1.00315, 1.00277, 1.00247,
      1.00003, 0.99994, 1.00116, 0.99952, 1.0005, 1.00091, 1.00213, 1.00019,
      0.99977, 1.0003, 0.99981, 1.00003, 0.99967, 1.00306, 1.00072, 0.9996,
      0.99979;
  Eigen::VectorXd rhat_tail(48);
  rhat_tail << 1.00063, 1.00497, 1.00918, 0.99965, 1.0015, 0.99962, 1.00776,
      1.00042, 0.99963, 0.99951, 0.99984, 1.00054, 1.00403, 1.00107, 1.00287,
      1.00627, 1.00134, 0.99957, 0.99997, 1.00368, 1.00053, 1.00133, 1.00589,
      1.00107, 1.00031, 1.00229, 1.0, 1.00008, 1.0001, 1.00116, 1.00219,
      0.99992, 1.001, 1.01267, 1.00011, 1.00066, 1.00065, 1.00237, 0.9995,
      1.00104, 1.00341, 0.99958, 1.00033, 0.9996, 0.99957, 1.00058, 1.00191,
      1.00658;

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

    double computed_bulk_rhat, computed_tail_rhat;
    std::tie(computed_bulk_rhat, computed_tail_rhat)
        = stan::analyze::compute_potential_scale_reduction_rank(draws, size);
    double expected_bulk_rhat = rhat_bulk(index - 4);
    double expected_tail_rhat = rhat_tail(index - 4);

    ASSERT_NEAR(expected_bulk_rhat, computed_bulk_rhat, 1e-4)
        << "Bulk Rhat mismatch for index: " << index
        << ", parameter: " << chains.param_name(index);
    ASSERT_NEAR(expected_tail_rhat, computed_tail_rhat, 1e-4)
        << "Tail Rhat mismatch for index: " << index
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

TEST_F(ComputeRhat, chains_compute_split_potential_scale_reduction_rank) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd rhat_bulk(48);
  rhat_bulk << 1.0078, 1.0109, 0.99919, 1.001, 1.00401, 1.00992, 1.00182,
      1.00519, 1.00095, 1.00351, 1.00554, 1.00075, 1.00595, 1.00473, 1.00546,
      1.01304, 1.00166, 1.0074, 1.00178, 1.00588, 1.00406, 1.00129, 1.00976,
      1.0013, 1.00193, 1.00104, 0.99938, 1.00025, 1.01082, 1.0019, 1.00354,
      1.0043, 1.00111, 1.00281, 1.00436, 1.00515, 1.00325, 1.0089, 1.00222,
      1.00118, 1.00191, 1.00283, 1.0003, 1.00216, 1.00335, 1.00133, 1.00023,
      1.0109;
  Eigen::VectorXd rhat_tail(48);
  rhat_tail << 1.00097, 1.00422, 1.00731, 1.00333, 1.00337, 0.99917, 1.00734,
      1.00633, 1.00074, 1.00906, 1.01019, 1.00074, 1.00447, 1.00383, 1.00895,
      1.00389, 1.00052, 1.00188, 1.00236, 1.00284, 1.00414, 1.00303, 1.00327,
      1.00295, 1.00037, 1.0044, 1.00488, 1.00178, 1.00475, 1.00082, 1.00413,
      1.01303, 1.0024, 1.01148, 1.00098, 1.00078, 1.00712, 1.00595, 1.00124,
      1.00112, 1.00381, 1.0006, 1.00188, 1.00225, 1.0026, 1.0009, 1.00209,
      1.00464;

  for (int index = 4; index < chains.num_params(); index++) {
    double computed_bulk_rhat, computed_tail_rhat;
    std::tie(computed_bulk_rhat, computed_tail_rhat)
        = chains.split_potential_scale_reduction_rank(index);
    double expected_bulk_rhat = rhat_bulk(index - 4);
    double expected_tail_rhat = rhat_tail(index - 4);

    ASSERT_NEAR(expected_bulk_rhat, computed_bulk_rhat, 1e-4)
        << "Bulk Rhat mismatch for index: " << index
        << ", parameter: " << chains.param_name(index);
    ASSERT_NEAR(expected_tail_rhat, computed_tail_rhat, 1e-4)
        << "Tail Rhat mismatch for index: " << index
        << ", parameter: " << chains.param_name(index);
  }
  for (int index = 0; index < chains.num_params(); index++) {
    std::string name = chains.param_name(index);
    ASSERT_EQ(chains.split_potential_scale_reduction_rank(index),
              chains.split_potential_scale_reduction_rank(name));
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

TEST_F(ComputeRhat, compute_split_potential_scale_reduction_rank) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);

  chains.add(blocker2);

  Eigen::VectorXd rhat_bulk(48);
  rhat_bulk << 1.0078, 1.0109, 0.99919, 1.001, 1.00401, 1.00992, 1.00182,
      1.00519, 1.00095, 1.00351, 1.00554, 1.00075, 1.00595, 1.00473, 1.00546,
      1.01304, 1.00166, 1.0074, 1.00178, 1.00588, 1.00406, 1.00129, 1.00976,
      1.0013, 1.00193, 1.00104, 0.99938, 1.00025, 1.01082, 1.0019, 1.00354,
      1.0043, 1.00111, 1.00281, 1.00436, 1.00515, 1.00325, 1.0089, 1.00222,
      1.00118, 1.00191, 1.00283, 1.0003, 1.00216, 1.00335, 1.00133, 1.00023,
      1.0109;
  Eigen::VectorXd rhat_tail(48);
  rhat_tail << 1.00097, 1.00422, 1.00731, 1.00333, 1.00337, 0.99917, 1.00734,
      1.00633, 1.00074, 1.00906, 1.01019, 1.00074, 1.00447, 1.00383, 1.00895,
      1.00389, 1.00052, 1.00188, 1.00236, 1.00284, 1.00414, 1.00303, 1.00327,
      1.00295, 1.00037, 1.0044, 1.00488, 1.00178, 1.00475, 1.00082, 1.00413,
      1.01303, 1.0024, 1.01148, 1.00098, 1.00078, 1.00712, 1.00595, 1.00124,
      1.00112, 1.00381, 1.0006, 1.00188, 1.00225, 1.0026, 1.0009, 1.00209,
      1.00464;

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

    double computed_bulk_rhat, computed_tail_rhat;
    std::tie(computed_bulk_rhat, computed_tail_rhat)
        = stan::analyze::compute_split_potential_scale_reduction_rank(draws,
                                                                      sizes);
    double expected_bulk_rhat = rhat_bulk(index - 4);
    double expected_tail_rhat = rhat_tail(index - 4);

    ASSERT_NEAR(expected_bulk_rhat, computed_bulk_rhat, 1e-4)
        << "Bulk Rhat mismatch for index: " << index
        << ", parameter: " << chains.param_name(index);
    ASSERT_NEAR(expected_tail_rhat, computed_tail_rhat, 1e-4)
        << "Tail Rhat mismatch for index: " << index
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

TEST_F(ComputeRhat, compute_split_potential_scale_reduction_convenience_rank) {
  std::stringstream out;
  stan::io::stan_csv blocker1
      = stan::io::stan_csv_reader::parse(blocker1_stream, &out);
  stan::io::stan_csv blocker2
      = stan::io::stan_csv_reader::parse(blocker2_stream, &out);
  EXPECT_EQ("", out.str());

  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd rhat_bulk(48);
  rhat_bulk << 1.0078, 1.0109, 0.99919, 1.001, 1.00401, 1.00992, 1.00182,
      1.00519, 1.00095, 1.00351, 1.00554, 1.00075, 1.00595, 1.00473, 1.00546,
      1.01304, 1.00166, 1.0074, 1.00178, 1.00588, 1.00406, 1.00129, 1.00976,
      1.0013, 1.00193, 1.00104, 0.99938, 1.00025, 1.01082, 1.0019, 1.00354,
      1.0043, 1.00111, 1.00281, 1.00436, 1.00515, 1.00325, 1.0089, 1.00222,
      1.00118, 1.00191, 1.00283, 1.0003, 1.00216, 1.00335, 1.00133, 1.00023,
      1.0109;
  Eigen::VectorXd rhat_tail(48);
  rhat_tail << 1.00097, 1.00422, 1.00731, 1.00333, 1.00337, 0.99917, 1.00734,
      1.00633, 1.00074, 1.00906, 1.01019, 1.00074, 1.00447, 1.00383, 1.00895,
      1.00389, 1.00052, 1.00188, 1.00236, 1.00284, 1.00414, 1.00303, 1.00327,
      1.00295, 1.00037, 1.0044, 1.00488, 1.00178, 1.00475, 1.00082, 1.00413,
      1.01303, 1.0024, 1.01148, 1.00098, 1.00078, 1.00712, 1.00595, 1.00124,
      1.00112, 1.00381, 1.0006, 1.00188, 1.00225, 1.0026, 1.0009, 1.00209,
      1.00464;

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

    double computed_bulk_rhat, computed_tail_rhat;
    std::tie(computed_bulk_rhat, computed_tail_rhat)
        = stan::analyze::compute_split_potential_scale_reduction_rank(draws,
                                                                      size);
    double expected_bulk_rhat = rhat_bulk(index - 4);
    double expected_tail_rhat = rhat_tail(index - 4);

    ASSERT_NEAR(expected_bulk_rhat, computed_bulk_rhat, 1e-4)
        << "Bulk Rhat mismatch for index: " << index
        << ", parameter: " << chains.param_name(index);
    ASSERT_NEAR(expected_tail_rhat, computed_tail_rhat, 1e-4)
        << "Tail Rhat mismatch for index: " << index
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

TEST_F(ComputeRhat, compute_potential_scale_reduction_rank_constant) {
  std::vector<std::string> param_names{"a"};
  stan::mcmc::chains<> chains(param_names);
  Eigen::Matrix<double, 2, 1> draws;
  draws << 1.0, 1.0;
  chains.add(draws);

  double computed_bulk_rhat, computed_tail_rhat;
  std::tie(computed_bulk_rhat, computed_tail_rhat)
      = chains.split_potential_scale_reduction_rank(0);

  ASSERT_TRUE(std::isnan(computed_bulk_rhat))
      << "rhat for index: " << 1 << ", parameter: " << chains.param_name(1);
  ASSERT_TRUE(std::isnan(computed_tail_rhat))
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

TEST_F(ComputeRhat, compute_potential_scale_reduction_rank_nan) {
  std::vector<std::string> param_names{"a"};
  stan::mcmc::chains<> chains(param_names);
  Eigen::Matrix<double, 2, 1> draws;
  draws << 1.0, std::numeric_limits<double>::quiet_NaN();
  chains.add(draws);

  double computed_bulk_rhat, computed_tail_rhat;
  std::tie(computed_bulk_rhat, computed_tail_rhat)
      = chains.split_potential_scale_reduction_rank(0);

  ASSERT_TRUE(std::isnan(computed_bulk_rhat))
      << "rhat for index: " << 1 << ", parameter: " << chains.param_name(1);
  ASSERT_TRUE(std::isnan(computed_tail_rhat))
      << "rhat for index: " << 1 << ", parameter: " << chains.param_name(1);
}
