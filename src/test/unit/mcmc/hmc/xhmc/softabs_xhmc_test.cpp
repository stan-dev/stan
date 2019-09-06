#include <stan/callbacks/stream_logger.hpp>
#include <stan/mcmc/hmc/xhmc/softabs_xhmc.hpp>
#include <boost/random/additive_combine.hpp>
#include <test/test-models/good/mcmc/hmc/common/gauss3D.hpp>
#include <stan/io/dump.hpp>
#include <fstream>

#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;

TEST(McmcUnitEXHMC, build_tree) {
  rng_t base_rng(4839294);

  stan::mcmc::softabs_point z_init(3);
  z_init.q(0) = 1;
  z_init.q(1) = -1;
  z_init.q(2) = 1;
  z_init.p(0) = -1;
  z_init.p(1) = 1;
  z_init.p(2) = -1;

  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  std::fstream empty_stream("", std::fstream::in);
  stan::io::dump data_var_context(empty_stream);
  gauss3D_model_namespace::gauss3D_model model(data_var_context);

  stan::mcmc::softabs_xhmc<gauss3D_model_namespace::gauss3D_model, rng_t>
    sampler(model, base_rng);

  sampler.z() = z_init;
  sampler.init_hamiltonian(logger);
  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();

  stan::mcmc::ps_point z_propose = z_init;

  double ave = 0;
  double log_sum_weight = -std::numeric_limits<double>::infinity();

  double H0 = -0.1;
  int n_leapfrog = 0;
  double sum_metro_prob = 0;

  bool valid_subtree = sampler.build_tree(3, z_propose, ave, log_sum_weight,
                                          H0, 1, n_leapfrog,
                                          sum_metro_prob, logger);

  EXPECT_EQ(0.1, sampler.get_nominal_stepsize());

  EXPECT_TRUE(valid_subtree);

  EXPECT_FLOAT_EQ(0.20423166, sampler.z().q(0));
  EXPECT_FLOAT_EQ(-0.20423166, sampler.z().q(1));
  EXPECT_FLOAT_EQ(0.20423166, sampler.z().q(2));

  EXPECT_FLOAT_EQ(-1.5019561, sampler.z().p(0));
  EXPECT_FLOAT_EQ(1.5019561, sampler.z().p(1));
  EXPECT_FLOAT_EQ(-1.5019561, sampler.z().p(2));

  EXPECT_FLOAT_EQ(0.8330583, z_propose.q(0));
  EXPECT_FLOAT_EQ(-0.8330583, z_propose.q(1));
  EXPECT_FLOAT_EQ(0.8330583, z_propose.q(2));

  EXPECT_FLOAT_EQ(-1.1836562, z_propose.p(0));
  EXPECT_FLOAT_EQ(1.1836562, z_propose.p(1));
  EXPECT_FLOAT_EQ(-1.1836562, z_propose.p(2));

  EXPECT_EQ(8, n_leapfrog);
  EXPECT_FLOAT_EQ(3.7645235, ave);
  EXPECT_FLOAT_EQ(std::log(0.34310558), log_sum_weight);
  EXPECT_FLOAT_EQ(0.34310558, sum_metro_prob);

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST(McmcUnitEXHMC, transition) {
  rng_t base_rng(4839294);

  stan::mcmc::softabs_point z_init(3);
  z_init.q(0) = 1;
  z_init.q(1) = -1;
  z_init.q(2) = 1;
  z_init.p(0) = -1;
  z_init.p(1) = 1;
  z_init.p(2) = -1;

  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  std::fstream empty_stream("", std::fstream::in);
  stan::io::dump data_var_context(empty_stream);
  gauss3D_model_namespace::gauss3D_model model(data_var_context);

  stan::mcmc::softabs_xhmc<gauss3D_model_namespace::gauss3D_model, rng_t>
    sampler(model, base_rng);

  sampler.z() = z_init;
  sampler.init_hamiltonian(logger);
  sampler.set_nominal_stepsize(0.1);
  sampler.set_stepsize_jitter(0);
  sampler.sample_stepsize();

  stan::mcmc::sample init_sample(z_init.q, 0, 0);

  stan::mcmc::sample s = sampler.transition(init_sample, logger);

  EXPECT_FLOAT_EQ(1, s.cont_params()(0));
  EXPECT_FLOAT_EQ(-1, s.cont_params()(1));
  EXPECT_FLOAT_EQ(1, s.cont_params()(2));
  EXPECT_FLOAT_EQ(-1.5, s.log_prob());
  EXPECT_FLOAT_EQ(0.99829924, s.accept_stat());

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}
