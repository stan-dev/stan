#include <stan/mcmc/hmc/static_uniform/softabs_static_uniform.hpp>
#include <stan/mcmc/hmc/static_uniform/unit_e_static_uniform.hpp>
#include <stan/mcmc/hmc/static_uniform/diag_e_static_uniform.hpp>
#include <stan/mcmc/hmc/static_uniform/dense_e_static_uniform.hpp>
#include <stan/mcmc/hmc/static_uniform/adapt_unit_e_static_uniform.hpp>
#include <stan/mcmc/hmc/static_uniform/adapt_diag_e_static_uniform.hpp>
#include <stan/mcmc/hmc/static_uniform/adapt_dense_e_static_uniform.hpp>
#include <stan/mcmc/hmc/static_uniform/adapt_softabs_static_uniform.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/services/util/create_rng.hpp>

#include <test/test-models/good/mcmc/hmc/common/gauss.hpp>

#include <gtest/gtest.h>

TEST(McmcStaticUniform, unit_e_transition) {
  stan::rng_t base_rng = stan::services::util::create_rng(4839294, 0);

  stan::mcmc::unit_e_point z_init(1);
  z_init.q(0) = 1;
  z_init.p(0) = -1;

  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  stan::io::empty_var_context data_var_context;
  gauss_model_namespace::gauss_model model(data_var_context);

  stan::mcmc::unit_e_static_uniform<gauss_model_namespace::gauss_model,
                                    stan::rng_t>
      sampler(model, base_rng);

  sampler.z() = z_init;
  sampler.init_hamiltonian(logger);
  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();

  stan::mcmc::sample init_sample(z_init.q, 0, 0);

  stan::mcmc::sample s = sampler.transition(init_sample, logger);

  EXPECT_FLOAT_EQ(1.0920367, s.cont_params()(0));
  EXPECT_FLOAT_EQ(-0.59627211, s.log_prob());
  EXPECT_FLOAT_EQ(0.99985325, s.accept_stat());
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST(McmcStaticUniform, diag_e_transition) {
  stan::rng_t base_rng = stan::services::util::create_rng(4839294, 0);

  stan::mcmc::diag_e_point z_init(1);
  z_init.q(0) = 1;
  z_init.p(0) = -1;

  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  stan::io::empty_var_context data_var_context;
  gauss_model_namespace::gauss_model model(data_var_context);

  stan::mcmc::diag_e_static_uniform<gauss_model_namespace::gauss_model,
                                    stan::rng_t>
      sampler(model, base_rng);

  sampler.z() = z_init;
  sampler.init_hamiltonian(logger);
  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();

  stan::mcmc::sample init_sample(z_init.q, 0, 0);

  stan::mcmc::sample s = sampler.transition(init_sample, logger);

  EXPECT_FLOAT_EQ(1.0920367, s.cont_params()(0));
  EXPECT_FLOAT_EQ(-0.59627211, s.log_prob());
  EXPECT_FLOAT_EQ(0.99985325, s.accept_stat());
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST(McmcStaticUniform, dense_e_transition) {
  stan::rng_t base_rng = stan::services::util::create_rng(4839294, 0);

  stan::mcmc::dense_e_point z_init(1);
  z_init.q(0) = 1;
  z_init.p(0) = -1;

  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  stan::io::empty_var_context data_var_context;
  gauss_model_namespace::gauss_model model(data_var_context);

  stan::mcmc::dense_e_static_uniform<gauss_model_namespace::gauss_model,
                                     stan::rng_t>
      sampler(model, base_rng);

  sampler.z() = z_init;
  sampler.init_hamiltonian(logger);
  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();

  stan::mcmc::sample init_sample(z_init.q, 0, 0);

  stan::mcmc::sample s = sampler.transition(init_sample, logger);

  EXPECT_FLOAT_EQ(1.0920367, s.cont_params()(0));
  EXPECT_FLOAT_EQ(-0.59627211, s.log_prob());
  EXPECT_FLOAT_EQ(0.99985325, s.accept_stat());
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST(McmcStaticUniform, softabs_transition) {
  stan::rng_t base_rng = stan::services::util::create_rng(4839294, 0);

  stan::mcmc::softabs_point z_init(1);
  z_init.q(0) = 1;
  z_init.p(0) = -1;

  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  stan::io::empty_var_context data_var_context;
  gauss_model_namespace::gauss_model model(data_var_context);

  stan::mcmc::softabs_static_uniform<gauss_model_namespace::gauss_model,
                                     stan::rng_t>
      sampler(model, base_rng);

  sampler.z() = z_init;
  sampler.init_hamiltonian(logger);
  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();

  stan::mcmc::sample init_sample(z_init.q, 0, 0);

  stan::mcmc::sample s = sampler.transition(init_sample, logger);

  EXPECT_FLOAT_EQ(1.0826443, s.cont_params()(0));
  EXPECT_FLOAT_EQ(-0.58605933, s.log_prob());
  EXPECT_FLOAT_EQ(0.99989599, s.accept_stat());
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST(McmcStaticUniform, adapt_unit_e_transition) {
  stan::rng_t base_rng = stan::services::util::create_rng(4839294, 0);

  stan::mcmc::unit_e_point z_init(1);
  z_init.q(0) = 1;
  z_init.p(0) = -1;

  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  stan::io::empty_var_context data_var_context;
  gauss_model_namespace::gauss_model model(data_var_context);

  stan::mcmc::adapt_unit_e_static_uniform<gauss_model_namespace::gauss_model,
                                          stan::rng_t>
      sampler(model, base_rng);

  sampler.z() = z_init;
  sampler.init_hamiltonian(logger);
  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();

  stan::mcmc::sample init_sample(z_init.q, 0, 0);

  stan::mcmc::sample s = sampler.transition(init_sample, logger);

  EXPECT_FLOAT_EQ(1.0920367, s.cont_params()(0));
  EXPECT_FLOAT_EQ(-0.59627211, s.log_prob());
  EXPECT_FLOAT_EQ(0.99985325, s.accept_stat());
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST(McmcStaticUniform, adapt_diag_e_transition) {
  stan::rng_t base_rng = stan::services::util::create_rng(4839294, 0);

  stan::mcmc::diag_e_point z_init(1);
  z_init.q(0) = 1;
  z_init.p(0) = -1;

  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  stan::io::empty_var_context data_var_context;
  gauss_model_namespace::gauss_model model(data_var_context);

  stan::mcmc::adapt_diag_e_static_uniform<gauss_model_namespace::gauss_model,
                                          stan::rng_t>
      sampler(model, base_rng);

  sampler.z() = z_init;
  sampler.init_hamiltonian(logger);
  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();

  stan::mcmc::sample init_sample(z_init.q, 0, 0);

  stan::mcmc::sample s = sampler.transition(init_sample, logger);

  EXPECT_FLOAT_EQ(1.0920367, s.cont_params()(0));
  EXPECT_FLOAT_EQ(-0.59627211, s.log_prob());
  EXPECT_FLOAT_EQ(0.99985325, s.accept_stat());
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST(McmcStaticUniform, adapt_dense_e_transition) {
  stan::rng_t base_rng = stan::services::util::create_rng(4839294, 0);

  stan::mcmc::dense_e_point z_init(1);
  z_init.q(0) = 1;
  z_init.p(0) = -1;

  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  stan::io::empty_var_context data_var_context;
  gauss_model_namespace::gauss_model model(data_var_context);

  stan::mcmc::adapt_dense_e_static_uniform<gauss_model_namespace::gauss_model,
                                           stan::rng_t>
      sampler(model, base_rng);

  sampler.z() = z_init;
  sampler.init_hamiltonian(logger);
  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();

  stan::mcmc::sample init_sample(z_init.q, 0, 0);

  stan::mcmc::sample s = sampler.transition(init_sample, logger);

  EXPECT_FLOAT_EQ(1.0920367, s.cont_params()(0));
  EXPECT_FLOAT_EQ(-0.59627211, s.log_prob());
  EXPECT_FLOAT_EQ(0.99985325, s.accept_stat());
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST(McmcStaticUniform, adapt_softabs_e_transition) {
  stan::rng_t base_rng = stan::services::util::create_rng(4839294, 0);

  stan::mcmc::softabs_point z_init(1);
  z_init.q(0) = 1;
  z_init.p(0) = -1;

  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  stan::io::empty_var_context data_var_context;
  gauss_model_namespace::gauss_model model(data_var_context);

  stan::mcmc::adapt_softabs_static_uniform<gauss_model_namespace::gauss_model,
                                           stan::rng_t>
      sampler(model, base_rng);

  sampler.z() = z_init;
  sampler.init_hamiltonian(logger);
  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();

  stan::mcmc::sample init_sample(z_init.q, 0, 0);

  stan::mcmc::sample s = sampler.transition(init_sample, logger);

  EXPECT_FLOAT_EQ(1.0826443, s.cont_params()(0));
  EXPECT_FLOAT_EQ(-0.58605933, s.log_prob());
  EXPECT_FLOAT_EQ(0.99989599, s.accept_stat());
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}
