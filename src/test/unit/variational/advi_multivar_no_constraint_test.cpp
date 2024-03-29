#include <test/test-models/good/variational/multivariate_no_constraint.hpp>
#include <stan/variational/advi.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/io/empty_var_context.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>
#include <string>
#include <stan/services/util/create_rng.hpp>

typedef multivariate_no_constraint_model_namespace::
    multivariate_no_constraint_model Model;

TEST(advi_test, multivar_no_constraint_fullrank) {
  stan::io::empty_var_context dummy_context;

  // Instantiate model
  Model my_model(dummy_context);

  // RNG
  stan::rng_t base_rng = stan::services::util::create_rng(0, 0);

  // Other params
  int n_monte_carlo_grad = 10;
  int n_grad_samples = 1e4;
  std::stringstream log_stream;
  stan::callbacks::stream_logger logger(log_stream, log_stream, log_stream,
                                        log_stream, log_stream);

  // Dummy input
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(2);
  cont_params(0) = 0.75;
  cont_params(1) = 0.75;

  // ADVI
  stan::variational::advi<Model, stan::variational::normal_fullrank,
                          stan::rng_t>
      test_advi(my_model, cont_params, base_rng, n_monte_carlo_grad,
                n_grad_samples, 100, 1);

  // Create some arbitrary variational q() family to calculate the ELBO over
  Eigen::VectorXd mu = Eigen::VectorXd::Constant(my_model.num_params_r(), 2.5);
  Eigen::MatrixXd L_chol = Eigen::MatrixXd::Identity(my_model.num_params_r(),
                                                     my_model.num_params_r());
  stan::variational::normal_fullrank muL
      = stan::variational::normal_fullrank(mu, L_chol);

  double elbo = 0.0;
  elbo = test_advi.calc_ELBO(muL, logger);

  // Can calculate ELBO analytically
  double zeta = -0.5 * (3 * 2 * log(2.0 * stan::math::pi()) + 18.5 + 25 + 13);
  Eigen::VectorXd mu_J = Eigen::VectorXd::Zero(2);
  mu_J(0) = 10.5;
  mu_J(1) = 7.5;

  double elbo_true = 0.0;
  elbo_true += zeta;
  elbo_true += mu_J.dot(mu);
  elbo_true += -0.5 * (3 * mu.dot(mu) + 3 * 2);
  elbo_true += 1 + log(2.0 * stan::math::pi());

  double const EPSILON = 0.1;
  EXPECT_NEAR(elbo_true, elbo, EPSILON);

  Eigen::VectorXd mu_grad = Eigen::VectorXd::Zero(3);
  Eigen::MatrixXd L_grad = Eigen::MatrixXd::Identity(my_model.num_params_r(),
                                                     my_model.num_params_r());

  std::string error
      = "stan::variational::normal_fullrank: "
        "Dimension of mean vector (3) and "
        "Dimension of Cholesky factor (2) must match in size";
  EXPECT_THROW_MSG(stan::variational::normal_fullrank(mu_grad, L_grad),
                   std::invalid_argument, error);

  mu_grad = Eigen::VectorXd::Zero(0);

  error
      = "stan::variational::normal_fullrank: "
        "Dimension of mean vector (0) and "
        "Dimension of Cholesky factor (2) must match in size";
  EXPECT_THROW_MSG(stan::variational::normal_fullrank(mu_grad, L_grad),
                   std::invalid_argument, error);

  mu_grad = Eigen::VectorXd::Zero(my_model.num_params_r());
  L_grad = Eigen::MatrixXd::Identity(3, 3);

  error
      = "stan::variational::normal_fullrank: "
        "Dimension of mean vector (2) and "
        "Dimension of Cholesky factor (3) must match in size";
  EXPECT_THROW_MSG(stan::variational::normal_fullrank(mu_grad, L_grad),
                   std::invalid_argument, error);

  mu_grad = Eigen::VectorXd::Zero(my_model.num_params_r());
  L_grad = Eigen::MatrixXd::Identity(0, 0);

  error
      = "stan::variational::normal_fullrank: "
        "Dimension of mean vector (2) and "
        "Dimension of Cholesky factor (0) must match in size";
  EXPECT_THROW_MSG(stan::variational::normal_fullrank(mu_grad, L_grad),
                   std::invalid_argument, error);

  mu_grad = Eigen::VectorXd::Zero(my_model.num_params_r());
  L_grad = Eigen::MatrixXd::Identity(1, 4);

  error
      = "stan::variational::normal_fullrank: "
        "Expecting a square matrix; rows of Cholesky factor (1) and columns "
        "of Cholesky factor (4) must match in size";
  EXPECT_THROW_MSG(stan::variational::normal_fullrank(mu_grad, L_grad),
                   std::invalid_argument, error);

  mu_grad = Eigen::VectorXd::Zero(3);
  L_grad = Eigen::MatrixXd::Identity(3, 3);
  stan::variational::normal_fullrank elbo_grad
      = stan::variational::normal_fullrank(mu_grad, L_grad);

  error
      = "stan::variational::normal_fullrank::calc_grad: "
        "Dimension of elbo_grad (3) and "
        "Dimension of variational q (2) must match in size";
  EXPECT_THROW_MSG(muL.calc_grad(elbo_grad, my_model, cont_params,
                                 n_monte_carlo_grad, base_rng, logger),
                   std::invalid_argument, error);
}

TEST(advi_test, multivar_no_constraint_meanfield) {
  stan::io::empty_var_context dummy_context;

  // Instantiate model
  Model my_model(dummy_context);

  // RNG
  stan::rng_t base_rng = stan::services::util::create_rng(0, 0);

  // Other params
  int n_monte_carlo_grad = 10;
  std::stringstream log_stream;
  stan::callbacks::stream_logger logger(log_stream, log_stream, log_stream,
                                        log_stream, log_stream);

  // Dummy input
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(2);
  cont_params(0) = 0.75;
  cont_params(1) = 0.75;

  // ADVI
  stan::variational::advi<Model, stan::variational::normal_meanfield,
                          stan::rng_t>
      test_advi(my_model, cont_params, base_rng, n_monte_carlo_grad,
                1e4,  // absurdly high!
                100, 1);

  // Create some arbitrary variational q() family to calculate the ELBO over
  Eigen::VectorXd mu = Eigen::VectorXd::Constant(my_model.num_params_r(), 2.5);
  Eigen::VectorXd sigma_tilde
      = Eigen::VectorXd::Constant(my_model.num_params_r(),
                                  0.0);  // initializing sigma_tilde = 0
                                         // means sigma = 1
  stan::variational::normal_meanfield musigmatilde
      = stan::variational::normal_meanfield(mu, sigma_tilde);

  double elbo = 0.0;
  elbo = test_advi.calc_ELBO(musigmatilde, logger);

  // Can calculate ELBO analytically
  double zeta = -0.5 * (3 * 2 * log(2.0 * stan::math::pi()) + 18.5 + 25 + 13);
  Eigen::VectorXd mu_J = Eigen::VectorXd::Zero(2);
  mu_J(0) = 10.5;
  mu_J(1) = 7.5;

  double elbo_true = 0.0;
  elbo_true += zeta;
  elbo_true += mu_J.dot(mu);
  elbo_true += -0.5 * (3 * mu.dot(mu) + 3 * 2);
  elbo_true += 1 + log(2.0 * stan::math::pi());

  double const EPSILON = 0.1;
  EXPECT_NEAR(elbo_true, elbo, EPSILON);

  Eigen::VectorXd mu_grad = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd st_grad = Eigen::VectorXd::Zero(my_model.num_params_r());

  std::string error
      = "stan::variational::normal_meanfield: "
        "Dimension of mean vector (3) and "
        "Dimension of log std vector (2) must match in size";
  EXPECT_THROW_MSG(stan::variational::normal_meanfield(mu_grad, st_grad),
                   std::invalid_argument, error);

  mu_grad = Eigen::VectorXd::Zero(0);

  error
      = "stan::variational::normal_meanfield: "
        "Dimension of mean vector (0) and "
        "Dimension of log std vector (2) must match in size";
  EXPECT_THROW_MSG(stan::variational::normal_meanfield(mu_grad, st_grad),
                   std::invalid_argument, error);

  mu_grad = Eigen::VectorXd::Zero(my_model.num_params_r());
  st_grad = Eigen::VectorXd::Zero(3);

  error
      = "stan::variational::normal_meanfield: "
        "Dimension of mean vector (2) and "
        "Dimension of log std vector (3) must match in size";
  EXPECT_THROW_MSG(stan::variational::normal_meanfield(mu_grad, st_grad),
                   std::invalid_argument, error);

  mu_grad = Eigen::VectorXd::Zero(my_model.num_params_r());
  st_grad = Eigen::VectorXd::Zero(0);

  error
      = "stan::variational::normal_meanfield: "
        "Dimension of mean vector (2) and "
        "Dimension of log std vector (0) must match in size";
  EXPECT_THROW_MSG(stan::variational::normal_meanfield(mu_grad, st_grad),
                   std::invalid_argument, error);

  mu_grad = Eigen::VectorXd::Zero(3);
  st_grad = Eigen::VectorXd::Zero(3);
  stan::variational::normal_meanfield elbo_grad
      = stan::variational::normal_meanfield(mu_grad, st_grad);

  error
      = "stan::variational::normal_meanfield::calc_grad: "
        "Dimension of elbo_grad (3) and "
        "Dimension of variational q (2) must match in size";
  EXPECT_THROW_MSG(musigmatilde.calc_grad(elbo_grad, my_model, cont_params,
                                          n_monte_carlo_grad, base_rng, logger),
                   std::invalid_argument, error);
}
