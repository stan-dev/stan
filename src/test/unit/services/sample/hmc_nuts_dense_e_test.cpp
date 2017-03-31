#include <stan/services/sample/hmc_nuts_dense_e_adapt.hpp>
#include <stan/services/sample/hmc_nuts_dense_e.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <iostream>

class ServicesSampleHmcNutsDenseE : public testing::Test {
public:
  ServicesSampleHmcNutsDenseE()
    : model(context, &model_log) {}

  std::stringstream model_log;
  stan::test::unit::instrumented_writer message, init, error;
  stan::test::unit::instrumented_writer parameter, diagnostic;
  stan::io::empty_var_context context;
  stan_model model;
};


TEST_F(ServicesSampleHmcNutsDenseE, call_count) {
  unsigned int random_seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int num_warmup = 200;
  int num_samples = 400;
  int num_thin = 5;
  bool save_warmup = true;
  int refresh = 0;
  double stepsize = 0.1;
  double stepsize_jitter = 0;
  int max_depth = 8;
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);
      
  int return_code = stan::services::sample::hmc_nuts_dense_e(
      model, context, random_seed, chain, init_radius,
      num_warmup, num_samples, num_thin, save_warmup, refresh,
      stepsize, stepsize_jitter, max_depth, 
      interrupt, message, error, init,
      parameter, diagnostic);
 
  EXPECT_EQ(0, return_code);

}


