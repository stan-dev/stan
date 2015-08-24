#include <test/test-models/good/variational/multivariate_with_constraint.hpp>
#include <stan/variational/advi.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG

typedef boost::ecuyer1988 rng_t;
typedef multivariate_with_constraint_model_namespace::multivariate_with_constraint_model Model;

TEST(advi_test, multivariate_with_constraint_test_adapt_eta) {
  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  std::stringstream print_stream_;
  std::stringstream output_stream_;
  std::stringstream diagnostic_stream_;

  print_stream_.str("");
  output_stream_.str("");
  diagnostic_stream_.str("");

  // Instantiate model
  Model my_model(dummy_context);

  // RNG
  rng_t base_rng(0);

  // Dummy input
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(my_model.num_params_r());

  // ADVI
  stan::variational::advi<Model, stan::variational::normal_fullrank, rng_t> test_advi(my_model,
                                                     cont_params,
                                                     10,
                                                     100,
                                                     base_rng,
                                                     100,
                                                     1,
                                                     &print_stream_,
                                                     &output_stream_,
                                                     &diagnostic_stream_);

  // ADVI should choose a relatively small eta and
  // should converge in < 2e4 iterations
  test_advi.run(0.0, 0.01, 2e4, 50);
}
