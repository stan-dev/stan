#include <test/test-models/good/variational/hier_logistic_cp.hpp>
#include <stan/variational/advi.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG

typedef boost::ecuyer1988 rng_t;
typedef hier_logistic_cp_model_namespace::hier_logistic_cp_model Model_cp;

TEST(advi_test, hier_logistic_cp_constraint_fullrank) {

  // Create mock data_var_context
  std::fstream data_stream("src/test/test-models/good/variational/hier_logistic.data.R",
                           std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  // Instantiate model
  Model_cp my_model(data_var_context);

  // RNG
  rng_t base_rng(0);

  // Dummy input
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(my_model.num_params_r());

  // ADVI
  stan::variational::advi<Model_cp, rng_t> test_advi(my_model,
                                                     cont_params,
                                                     0,
                                                     10,
                                                     10, // absurdly high!
                                                     0.01,
                                                     base_rng,
                                                     10,
                                                     &std::cout,
                                                     &std::cout,
                                                     &std::cout);

  test_advi.run_fullrank(1,1e4);
}

TEST(advi_test, hier_logistic_cp_constraint_meanfield) {

  // Create mock data_var_context
  std::fstream data_stream("src/test/test-models/good/variational/hier_logistic.data.R",
                           std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  // Instantiate model
  Model_cp my_model(data_var_context);

  // RNG
  rng_t base_rng(0);

  // Dummy input
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(my_model.num_params_r());

  // ADVI
  stan::variational::advi<Model_cp, rng_t> test_advi(my_model,
                                                     cont_params,
                                                     0,
                                                     10,
                                                     10, // absurdly high!
                                                     0.01,
                                                     base_rng,
                                                     10,
                                                     &std::cout,
                                                     &std::cout,
                                                     &std::cout);

  test_advi.run_meanfield(1,1e4);
}
