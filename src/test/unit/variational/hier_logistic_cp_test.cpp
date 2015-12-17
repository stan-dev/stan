#include <test/test-models/good/variational/hier_logistic_cp.hpp>
#include <stan/variational/advi.hpp>
#include <stan/interface_callbacks/writer/noop_writer.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG

typedef boost::ecuyer1988 rng_t;
typedef hier_logistic_cp_model_namespace::hier_logistic_cp_model Model_cp;

TEST(advi_test, hier_logistic_cp_constraint_meanfield) {
  // Create mock data_var_context
  std::fstream data_stream("src/test/test-models/good/variational/hier_logistic.data.R",
                           std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream output;
  output.clear();

  // Instantiate model
  Model_cp my_model(data_var_context);

  // RNG
  rng_t base_rng(0);

  // Dummy input
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(my_model.num_params_r());

  // ADVI
  stan::variational::advi<Model_cp, stan::variational::normal_meanfield, rng_t> test_advi(my_model,
                                                     cont_params,
                                                     base_rng,
                                                     10,
                                                     100,
                                                     100,
                                                     1);

  stan::interface_callbacks::writer::noop_writer writer;
  
  test_advi.run(0.01, false, 50, 1, 2e4,
                writer, writer, writer);
}
