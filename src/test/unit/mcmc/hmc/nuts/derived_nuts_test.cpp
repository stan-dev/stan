#include <stan/mcmc/hmc/nuts/softabs_nuts.hpp>
#include <stan/mcmc/hmc/nuts/unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/dense_e_nuts.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>
#include <stan/services/util/create_rng.hpp>

#include <test/unit/mcmc/hmc/mock_hmc.hpp>

#include <gtest/gtest.h>

TEST(McmcNutsDerivedNuts, compute_criterion_unit_e) {
  stan::rng_t base_rng = stan::services::util::create_rng(0, 0);

  int model_size = 1;

  stan::mcmc::unit_e_point start(model_size);
  stan::mcmc::unit_e_point finish(model_size);
  Eigen::VectorXd p_sharp_start(model_size);
  Eigen::VectorXd p_sharp_finish(model_size);
  Eigen::VectorXd rho(model_size);

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::unit_e_nuts<stan::mcmc::mock_model, stan::rng_t> sampler(
      model, base_rng);

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
  stan::rng_t base_rng = stan::services::util::create_rng(0, 0);

  int model_size = 1;

  stan::mcmc::diag_e_point start(model_size);
  stan::mcmc::diag_e_point finish(model_size);
  Eigen::VectorXd p_sharp_start(model_size);
  Eigen::VectorXd p_sharp_finish(model_size);
  Eigen::VectorXd rho(model_size);

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::diag_e_nuts<stan::mcmc::mock_model, stan::rng_t> sampler(
      model, base_rng);

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = 1;

  p_sharp_start = start.inv_e_metric_.cwiseProduct(start.p);
  p_sharp_finish = finish.inv_e_metric_.cwiseProduct(finish.p);
  rho = start.p + finish.p;

  EXPECT_TRUE(sampler.compute_criterion(p_sharp_start, p_sharp_finish, rho));

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = -1;

  p_sharp_start = start.inv_e_metric_.cwiseProduct(start.p);
  p_sharp_finish = finish.inv_e_metric_.cwiseProduct(finish.p);
  rho = start.p + finish.p;

  EXPECT_FALSE(sampler.compute_criterion(p_sharp_start, p_sharp_finish, rho));
}

TEST(McmcNutsDerivedNuts, compute_criterion_dense_e) {
  stan::rng_t base_rng = stan::services::util::create_rng(0, 0);

  int model_size = 1;

  stan::mcmc::dense_e_point start(model_size);
  stan::mcmc::dense_e_point finish(model_size);
  Eigen::VectorXd p_sharp_start(model_size);
  Eigen::VectorXd p_sharp_finish(model_size);
  Eigen::VectorXd rho(model_size);

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::dense_e_nuts<stan::mcmc::mock_model, stan::rng_t> sampler(
      model, base_rng);

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = 1;

  p_sharp_start = start.inv_e_metric_ * start.p;
  p_sharp_finish = finish.inv_e_metric_ * finish.p;
  rho = start.p + finish.p;

  EXPECT_TRUE(sampler.compute_criterion(p_sharp_start, p_sharp_finish, rho));

  start.q(0) = 1;
  start.p(0) = 1;

  finish.q(0) = 2;
  finish.p(0) = -1;

  p_sharp_start = start.inv_e_metric_ * start.p;
  p_sharp_finish = finish.inv_e_metric_ * finish.p;
  rho = start.p + finish.p;

  EXPECT_FALSE(sampler.compute_criterion(p_sharp_start, p_sharp_finish, rho));
}

TEST(McmcNutsDerivedNuts, compute_criterion_softabs) {
  stan::rng_t base_rng = stan::services::util::create_rng(0, 0);

  int model_size = 1;

  stan::mcmc::softabs_point start(model_size);
  stan::mcmc::softabs_point finish(model_size);
  Eigen::VectorXd p_sharp_start(model_size);
  Eigen::VectorXd p_sharp_finish(model_size);
  Eigen::VectorXd rho(model_size);

  stan::mcmc::mock_model model(model_size);
  stan::mcmc::softabs_nuts<stan::mcmc::mock_model, stan::rng_t> sampler(
      model, base_rng);

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
