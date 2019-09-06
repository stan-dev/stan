#include <gtest/gtest.h>

#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>
#include <stan/mcmc/hmc/nuts_classic/unit_e_nuts_classic.hpp>
#include <stan/mcmc/hmc/nuts_classic/diag_e_nuts_classic.hpp>
#include <stan/mcmc/hmc/nuts_classic/dense_e_nuts_classic.hpp>
#include <test/unit/mcmc/hmc/mock_hmc.hpp>
#include <boost/random/additive_combine.hpp>

typedef boost::ecuyer1988 rng_t;

TEST(McmcDerivedNutsClassic, compute_criterion_unit_e) {

  rng_t base_rng(0);

  int model_size = 1;

  stan::mcmc::ps_point start(model_size);
  stan::mcmc::unit_e_point finish(model_size);
  Eigen::VectorXd rho(model_size);

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::unit_e_nuts_classic<stan::mcmc::mock_model, rng_t>
    sampler(model, base_rng);

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = 1;

  rho = start.p + finish.p;

  EXPECT_EQ(true, sampler.compute_criterion(start, finish, rho));

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = -1;

  rho = start.p + finish.p;

  EXPECT_FALSE(sampler.compute_criterion(start, finish, rho));

}

TEST(McmcDerivedNutsClassic, compute_criterion_diag_e) {

  rng_t base_rng(0);

  int model_size = 1;

  stan::mcmc::ps_point start(model_size);
  stan::mcmc::diag_e_point finish(model_size);
  Eigen::VectorXd rho(model_size);

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::diag_e_nuts_classic<stan::mcmc::mock_model, rng_t>
    sampler(model, base_rng);

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = 1;

  rho = start.p + finish.p;

  EXPECT_EQ(true, sampler.compute_criterion(start, finish, rho));

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = -1;

  rho = start.p + finish.p;

  EXPECT_FALSE(sampler.compute_criterion(start, finish, rho));
}

TEST(McmcDerivedNutsClassic, compute_criterion_dense_e) {

  rng_t base_rng(0);

  int model_size = 1;

  stan::mcmc::ps_point start(model_size);
  stan::mcmc::dense_e_point finish(model_size);
  Eigen::VectorXd rho(model_size);

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::dense_e_nuts_classic<stan::mcmc::mock_model, rng_t>
    sampler(model, base_rng);

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = 1;

  rho = start.p + finish.p;

  EXPECT_EQ(true, sampler.compute_criterion(start, finish, rho));

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = -1;

  rho = start.p + finish.p;

  EXPECT_FALSE(sampler.compute_criterion(start, finish, rho));

}
