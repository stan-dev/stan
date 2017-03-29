#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/mcmc/hmc/common/gauss3D.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <iostream>
#include <boost/algorithm/string.hpp>

/** 
 * Use model with 3 params, fix seed, set mass matrix
 */

class ServicesSampleHmcNutsDiagEMassMatrix : public testing::Test {
public:
  ServicesSampleHmcNutsDiagEMassMatrix()
    : model(context, &model_log) {}

  std::stringstream model_log;
  stan::test::unit::instrumented_writer message, init, error;
  stan::test::unit::instrumented_writer parameter, diagnostic;
  stan::io::empty_var_context context;
  stan_model model;
};


TEST_F(ServicesSampleHmcNutsDiagEMassMatrix, no_adapt) {
  unsigned int random_seed = 12345;
  unsigned int chain = 1;
  double init_radius = 2;
  int num_warmup = 0;
  int num_samples = 2;
  int num_thin = 1;
  bool save_warmup = false;
  int refresh = 0;
  double stepsize = 1;
  double stepsize_jitter = 0;
  int max_depth = 10;
  double delta = .8;
  double gamma = .05;
  double kappa = .75;
  double t0 = 10;
  unsigned int init_buffer = 75;
  unsigned int term_buffer = 50;
  unsigned int window = 25;
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);

  std::string txt =
    "mass_matrix <- structure(c(0.787405, 0.884987, 1.19869),.Dim=c(3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  stan::io::var_context& inv_mass_matrix = dump;
  inv_mass_matrix.validate_dims("inv mass matrix", "mass_matrix",
                                "vector_d",
                                inv_mass_matrix.to_vec(3));
  std::vector<double> mass_matrix(3);
  mass_matrix = inv_mass_matrix.vals_r("mass_matrix");

  int return_code =
    stan::services::sample::hmc_nuts_diag_e_adapt(
    model, context, inv_mass_matrix, random_seed, chain, init_radius,
    num_warmup, num_samples, num_thin, save_warmup, refresh,
    stepsize, stepsize_jitter, max_depth, delta, gamma, kappa, t0,
    init_buffer, term_buffer, window,
    interrupt, message, error, init,
    parameter, diagnostic);
  EXPECT_EQ(0, return_code);

  // check returned mass matrix
  std::vector<std::string> param_strings = parameter.string_values();
  std::string vals;
  for (size_t i = 0; i < param_strings.size(); i++) {
    if (param_strings[i].find("Diagonal elements of inverse mass matrix:")
        != std::string::npos) {
      vals = param_strings[i+1];
      break;
    }
  }
  std::vector<std::string> strs;
  boost::split(strs, vals, boost::is_any_of(", "), boost::token_compress_on);
  EXPECT_EQ(3, strs.size());
  ASSERT_NEAR(0.78, std::stod(strs[0]), 0.05);
  ASSERT_NEAR(0.88, std::stod(strs[1]), 0.05);
  ASSERT_NEAR(1.20, std::stod(strs[2]), 0.05);

}


// run model for 2000 add'l iterations, starting from 250
TEST_F(ServicesSampleHmcNutsDiagEMassMatrix, continue_250_seed_123345) {
  unsigned int random_seed = 12345;   // not same result as running 2250 with this seed
  unsigned int chain = 1;
  double init_radius = 2;
  int num_warmup = 2000;
  int num_samples = 0;
  int num_thin = 5;
  bool save_warmup = false;
  int refresh = 0;
  double stepsize = 0.761;   // stepsize after 250 warmups
  double stepsize_jitter = 0;
  int max_depth = 10;
  double delta = .8;
  double gamma = .05;
  double kappa = .75;
  double t0 = 10;
  unsigned int init_buffer = 75;
  unsigned int term_buffer = 50;
  unsigned int window = 25;
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);

  // starting point from 250 warmups 
  std::string txt =
    "mass_matrix <- structure(c(0.787405, 0.884987, 1.19869),.Dim=c(3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  stan::io::var_context& inv_mass_matrix = dump;

  inv_mass_matrix.validate_dims("inv mass matrix", "mass_matrix",
                                "vector_d",
                                inv_mass_matrix.to_vec(3));
  std::vector<double> mass_matrix(3);
  mass_matrix = inv_mass_matrix.vals_r("mass_matrix");

  int return_code =
    stan::services::sample::hmc_nuts_diag_e_adapt(
    model, context, inv_mass_matrix, random_seed, chain, init_radius,
    num_warmup, num_samples, num_thin, save_warmup, refresh,
    stepsize, stepsize_jitter, max_depth, delta, gamma, kappa, t0,
    init_buffer, term_buffer, window,
    interrupt, message, error, init,
    parameter, diagnostic);
  EXPECT_EQ(0, return_code);

  std::vector<std::string> param_strings = parameter.string_values();
  std::string vals;
  for (size_t i = 0; i < param_strings.size(); i++) {
    if (param_strings[i].find("Diagonal elements of inverse mass matrix:")
        != std::string::npos) {
      vals = param_strings[i+1];
      break;
    }
  }
  std::vector<std::string> strs;
  boost::split(strs, vals, boost::is_any_of(", "), boost::token_compress_on);
  EXPECT_EQ(3, strs.size());
  ASSERT_NEAR(1.00, std::stod(strs[0]), 0.05);
  ASSERT_NEAR(1.00, std::stod(strs[1]), 0.05);
  ASSERT_NEAR(1.00, std::stod(strs[2]), 0.05);

}
