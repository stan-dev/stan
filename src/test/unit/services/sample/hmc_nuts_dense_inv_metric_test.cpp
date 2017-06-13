#include <stan/services/sample/hmc_nuts_dense_e_adapt.hpp>
#include <stan/services/sample/hmc_nuts_dense_e.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/mcmc/hmc/common/gauss3D.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/services/check_adaptation.hpp>
#include <iostream>
#include <gtest/gtest.h>

/**
 * Use 3-param model test-models/good/mcmc/hmc/common/gauss3D
 * fix seed 12345, test against specified inv Euclidean metric
 * Tests crafted by running samplers with test config
 * to capture resulting inverse Euclidean metric values.
 */

class ServicesSampleHmcNutsDenseEMassMatrix : public testing::Test {
public:
  ServicesSampleHmcNutsDenseEMassMatrix()
    : model(context, &model_log) {}

  std::stringstream model_log;
  stan::test::unit::instrumented_logger logger;
  stan::test::unit::instrumented_writer init, parameter, diagnostic;
  stan::io::empty_var_context context;
  stan_model model;
};

TEST_F(ServicesSampleHmcNutsDenseEMassMatrix, ident_no_adapt) {
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
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);

  int return_code =
    stan::services::sample::hmc_nuts_dense_e(model,
                                             context,
                                             random_seed,
                                             chain,
                                             init_radius,
                                             num_warmup,
                                             num_samples,
                                             num_thin,
                                             save_warmup,
                                             refresh,
                                             stepsize,
                                             stepsize_jitter,
                                             max_depth,
                                             interrupt,
                                             logger,
                                             init,
                                             parameter,
                                             diagnostic);
  EXPECT_EQ(0, return_code);

  stan::io::dump dmp =
    stan::services::util::create_unit_e_dense_inv_metric(3);
  stan::io::var_context& inv_metric = dmp;
  std::vector<double> dense_vals
    = inv_metric.vals_r("inv_metric");
  // check returned Euclidean metric
  stan::test::unit::check_adaptation(3, 3, dense_vals, parameter, 0.1);
}

TEST_F(ServicesSampleHmcNutsDenseEMassMatrix, ident_adapt_250) {
  unsigned int random_seed = 12345;
  unsigned int chain = 1;
  double init_radius = 2;
  int num_warmup = 250;
  int num_samples = 0;
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

  // run sampler with ident Euclidean metric
  int return_code =
    stan::services::sample::hmc_nuts_dense_e_adapt(model,
                                                   context,
                                                   random_seed,
                                                   chain,
                                                   init_radius,
                                                   num_warmup,
                                                   num_samples,
                                                   num_thin,
                                                   save_warmup,
                                                   refresh,
                                                   stepsize,
                                                   stepsize_jitter,
                                                   max_depth,
                                                   delta,
                                                   gamma,
                                                   kappa,
                                                   t0,
                                                   init_buffer,
                                                   term_buffer,
                                                   window,
                                                   interrupt,
                                                   logger,
                                                   init,
                                                   parameter,
                                                   diagnostic);
  EXPECT_EQ(0, return_code);
}

TEST_F(ServicesSampleHmcNutsDenseEMassMatrix, use_metric_no_adapt) {
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
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);

  std::string txt =
    "inv_metric <- structure(c("
    "0.640211, 0.156096, -0.374048, "
    "0.156096, 1.41239, -0.0412753, "
    "-0.374048, -0.0412753, 1.29567 "
    "), .Dim  = c(3,3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  stan::io::var_context& inv_metric = dump;

  int return_code =
    stan::services::sample::hmc_nuts_dense_e(model,
                                             context,
                                             inv_metric,
                                             random_seed,
                                             chain,
                                             init_radius,
                                             num_warmup,
                                             num_samples,
                                             num_thin,
                                             save_warmup,
                                             refresh,
                                             stepsize,
                                             stepsize_jitter,
                                             max_depth,
                                             interrupt,
                                             logger,
                                             init,
                                             parameter,
                                             diagnostic);
  EXPECT_EQ(0, return_code);

  std::vector<double> dense_vals(9);
  dense_vals = inv_metric.vals_r("inv_metric");
  stan::test::unit::check_adaptation(3, 3, dense_vals, parameter, 0.4);
}

TEST_F(ServicesSampleHmcNutsDenseEMassMatrix, use_metric_skip_adapt) {
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
    "inv_metric <- structure(c("
    "0.640211, 0.156096, -0.374048, "
    "0.156096, 1.41239, -0.0412753, "
    "-0.374048, -0.0412753, 1.29567 "
    "), .Dim  = c(3,3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  stan::io::var_context& inv_metric = dump;

  int return_code =
    stan::services::sample::hmc_nuts_dense_e_adapt(model,
                                                   context,
                                                   inv_metric,
                                                   random_seed,
                                                   chain,
                                                   init_radius,
                                                   num_warmup,
                                                   num_samples,
                                                   num_thin,
                                                   save_warmup,
                                                   refresh,
                                                   stepsize,
                                                   stepsize_jitter,
                                                   max_depth,
                                                   delta,
                                                   gamma,
                                                   kappa,
                                                   t0,
                                                   init_buffer,
                                                   term_buffer,
                                                   window,
                                                   interrupt,
                                                   logger,
                                                   init,
                                                   parameter,
                                                   diagnostic);

  EXPECT_EQ(0, return_code);

  std::vector<double> dense_vals(9);
  dense_vals = inv_metric.vals_r("inv_metric");
  stan::test::unit::check_adaptation(3, 3, dense_vals, parameter, 0.4);
}
