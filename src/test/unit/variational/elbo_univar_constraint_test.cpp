#include <test/test-models/good/variational/univar_constraint.hpp>
#include <stan/variational/advi.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>
#include <string>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG

typedef boost::ecuyer1988 rng_t;
typedef univar_constraint_model_namespace::univar_constraint_model Model;

TEST(advi_test, elbo_univar_contraint_fullrank) {
  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  Model my_model(dummy_context);
  rng_t base_rng(0);
  int n_monte_carlo_grad = 10;
  std::ostream* print_stream = &std::cout;
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(1);
  cont_params(0) = -0.75;

  stan::variational::advi<Model, stan::variational::normal_fullrank, rng_t>
    test_advi(my_model,
              cont_params,
              base_rng,
              n_monte_carlo_grad,
              5e5, // absurdly high!
              0.1,
              100,
              1,
              print_stream,
              &std::cout,
              &std::cout);

  // Create arbitrary variational family to calculate ELBO
  Eigen::VectorXd mu     = Eigen::VectorXd::Constant(my_model.num_params_r(),
                                                     log(1.88));
  Eigen::MatrixXd L_chol = Eigen::MatrixXd::Identity(my_model.num_params_r(),
                                                     my_model.num_params_r());
  stan::variational::normal_fullrank muL =
    stan::variational::normal_fullrank(mu, L_chol);

  double elbo = 0.0;
  elbo = test_advi.calc_ELBO(muL);

  // Calculate ELBO analytically + moment generating function of Gaussians
  double one_over_sigma_j_sq = 1.0 + 2*1.0;
  double sigma_j_sq = 1.0/one_over_sigma_j_sq;

  double mu_j = (1.5/1.0 + 1.6/1.0 + 1.4/1.0) * sigma_j_sq;

  double S_j = 1.0/(2.0*stan::math::pi())
               * sqrt( sigma_j_sq / (1.0) )
               * exp( -0.5 * ( pow(1.5,2)
                               + pow(1.4,2)
                               + pow(1.6,2)
                               - pow(mu_j,2)/sigma_j_sq ) );

  double elbo_true = 0.0;

  elbo_true += log(S_j);
  elbo_true += log(1.0/( sqrt( sigma_j_sq * 2.0 * stan::math::pi() ) ));
  elbo_true += -0.5*one_over_sigma_j_sq * ( exp(2.0*log(1.88)) * exp(2.0*1.0) );
  elbo_true += -0.5*one_over_sigma_j_sq * ( -2.0*mu_j*exp(log(1.88))*exp(0.5*1.0) );
  elbo_true += -0.5*one_over_sigma_j_sq * ( pow(mu_j, 2.0) );
  elbo_true += log(1.88);
  elbo_true += 0.5 * (1 + log(2.0*stan::math::pi()));

  double const EPSILON = 1.0;
  EXPECT_NEAR(elbo_true, elbo, EPSILON);
}

TEST(advi_test, elbo_univar_contraint_meanfield) {
  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  Model my_model(dummy_context);
  rng_t base_rng(0);
  int n_monte_carlo_grad = 10;
  std::ostream* print_stream = &std::cout;
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(1);
  cont_params(0) = -0.75;

  stan::variational::advi<Model, stan::variational::normal_meanfield, rng_t>
    test_advi(my_model,
              cont_params,
              base_rng,
              n_monte_carlo_grad,
              5e5, // absurdly high!
              0.1,
              100,
              1,
              print_stream,
              &std::cout,
              &std::cout);

  // Create arbitrary variational family to calculate ELBO
  Eigen::VectorXd mu  = Eigen::VectorXd::Constant(my_model.num_params_r(),
                                                     log(1.88));
  Eigen::MatrixXd sigma_tilde  = Eigen::VectorXd::Constant(
                                          my_model.num_params_r(),
                                          0.0); // initializing sigma_tilde = 0
                                                // means sigma = 1
  stan::variational::normal_meanfield musigmatilde =
    stan::variational::normal_meanfield(mu, sigma_tilde);

  double elbo = 0.0;
  elbo = test_advi.calc_ELBO(musigmatilde);

  // Calculate ELBO analytically + moment generating function of Gaussians
  double one_over_sigma_j_sq = 1.0 + 2*1.0;
  double sigma_j_sq = 1.0/one_over_sigma_j_sq;

  double mu_j = (1.5/1.0 + 1.6/1.0 + 1.4/1.0) * sigma_j_sq;

  double S_j = 1.0/(2.0*stan::math::pi())
               * sqrt( sigma_j_sq / (1.0) )
               * exp( -0.5 * ( pow(1.5,2)
                               + pow(1.4,2)
                               + pow(1.6,2)
                               - pow(mu_j,2)/sigma_j_sq ) );

  double elbo_true = 0.0;

  elbo_true += log(S_j);
  elbo_true += log(1.0/( sqrt( sigma_j_sq * 2.0 * stan::math::pi() ) ));
  elbo_true += -0.5*one_over_sigma_j_sq * ( exp(2.0*log(1.88)) * exp(2.0*1.0) );
  elbo_true += -0.5*one_over_sigma_j_sq * ( -2.0*mu_j*exp(log(1.88))*exp(0.5*1.0) );
  elbo_true += -0.5*one_over_sigma_j_sq * ( pow(mu_j, 2.0) );
  elbo_true += log(1.88);
  elbo_true += 0.5 * (1 + log(2.0*stan::math::pi()));


  double const EPSILON = 1.0;
  EXPECT_NEAR(elbo_true, elbo, EPSILON);
}
