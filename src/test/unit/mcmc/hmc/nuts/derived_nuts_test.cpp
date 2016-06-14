#include <stan/interface_callbacks/writer/noop_writer.hpp>
#include <stan/mcmc/hmc/nuts/softabs_nuts.hpp>
#include <stan/mcmc/hmc/nuts/unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/dense_e_nuts.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>

#include <test/unit/mcmc/hmc/mock_hmc.hpp>

#include <boost/random/additive_combine.hpp>

#include <gtest/gtest.h>


typedef boost::ecuyer1988 rng_t;

TEST(McmcNutsDerivedNuts, compute_criterion_unit_e) {

  rng_t base_rng(0);

  int model_size = 1;

  stan::mcmc::unit_e_point start(model_size);
  stan::mcmc::unit_e_point finish(model_size);
  Eigen::VectorXd p_sharp_start(model_size);
  Eigen::VectorXd p_sharp_finish(model_size);
  Eigen::VectorXd rho(model_size);

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::unit_e_nuts<stan::mcmc::mock_model, rng_t> sampler(model, base_rng);

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = 1;

  p_sharp_start = start.p;
  p_sharp_finish = finish.p;
  rho = start.p + finish.p;

  EXPECT_TRUE(sampler.compute_criterion(p_sharp_start, p_sharp_finish, rho));

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = -1;

  p_sharp_start = start.p;
  p_sharp_finish = finish.p;
  rho = start.p + finish.p;

  EXPECT_FALSE(sampler.compute_criterion(p_sharp_start, p_sharp_finish, rho));
}

TEST(McmcNutsDerivedNuts, compute_criterion_diag_e) {

  rng_t base_rng(0);

  int model_size = 1;

  stan::mcmc::diag_e_point start(model_size);
  stan::mcmc::diag_e_point finish(model_size);
  Eigen::VectorXd p_sharp_start(model_size);
  Eigen::VectorXd p_sharp_finish(model_size);
  Eigen::VectorXd rho(model_size);

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::diag_e_nuts<stan::mcmc::mock_model, rng_t> sampler(model, base_rng);

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = 1;

  p_sharp_start = start.mInv.cwiseProduct(start.p);
  p_sharp_finish = finish.mInv.cwiseProduct(finish.p);
  rho = start.p + finish.p;

  EXPECT_TRUE(sampler.compute_criterion(p_sharp_start, p_sharp_finish, rho));

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = -1;

  p_sharp_start = start.mInv.cwiseProduct(start.p);
  p_sharp_finish = finish.mInv.cwiseProduct(finish.p);
  rho = start.p + finish.p;

  EXPECT_FALSE(sampler.compute_criterion(p_sharp_start, p_sharp_finish, rho));
}

TEST(McmcNutsDerivedNuts, compute_criterion_dense_e) {

  rng_t base_rng(0);

  int model_size = 1;

  stan::mcmc::dense_e_point start(model_size);
  stan::mcmc::dense_e_point finish(model_size);
  Eigen::VectorXd p_sharp_start(model_size);
  Eigen::VectorXd p_sharp_finish(model_size);
  Eigen::VectorXd rho(model_size);

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::dense_e_nuts<stan::mcmc::mock_model, rng_t> sampler(model, base_rng);

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = 1;

  p_sharp_start = start.mInv * start.p;
  p_sharp_finish = finish.mInv * finish.p;
  rho = start.p + finish.p;

  EXPECT_TRUE(sampler.compute_criterion(p_sharp_start, p_sharp_finish, rho));

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = -1;

  p_sharp_start = start.mInv * start.p;
  p_sharp_finish = finish.mInv * finish.p;
  rho = start.p + finish.p;

  EXPECT_FALSE(sampler.compute_criterion(p_sharp_start, p_sharp_finish, rho));
}

TEST(McmcNutsDerivedNuts, compute_criterion_softabs) {

  rng_t base_rng(0);

  int model_size = 1;

  stan::mcmc::softabs_point start(model_size);
  stan::mcmc::softabs_point finish(model_size);
  Eigen::VectorXd p_sharp_start(model_size);
  Eigen::VectorXd p_sharp_finish(model_size);
  Eigen::VectorXd rho(model_size);

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::softabs_nuts<stan::mcmc::mock_model, rng_t>
    sampler(model, base_rng);

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = 1;

  p_sharp_start = start.p;
  p_sharp_finish = finish.p;
  rho = start.p + finish.p;

  EXPECT_TRUE(sampler.compute_criterion(p_sharp_start, p_sharp_finish, rho));

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = -1;

  p_sharp_start = start.p;
  p_sharp_finish = finish.p;
  rho = start.p + finish.p;

  EXPECT_FALSE(sampler.compute_criterion(p_sharp_start, p_sharp_finish, rho));
}
