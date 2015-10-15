#if 0
#include <test/test-models/good/variational/univar.hpp>
#include <test/test-models/good/variational/univar_constraint.hpp>
#include <test/test-models/good/variational/multivar.hpp>
#include <test/test-models/good/variational/multivar_constraint.hpp>
#include <stan/variational/advi.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>
#include <string>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG

typedef boost::ecuyer1988 rng_t;
typedef univar_model_namespace::univar_model Model_u;
typedef univar_constraint_model_namespace::univar_constraint_model Model_uc;
typedef multivar_model_namespace::multivar_model Model_m;
typedef multivar_constraint_model_namespace::multivar_constraint_model Model_mc;

TEST(AdviCalcElbo, univar_normal_fullrank) {
  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  Model_u my_model(dummy_context);
  rng_t base_rng(0);
  int n_monte_carlo_grad = 10;
  std::ostream* print_stream = &std::cout;
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(1);
  cont_params(0) = -0.75;

  stan::variational::advi<Model_u, stan::variational::normal_fullrank, rng_t>
    test_advi(my_model,
              cont_params,
              base_rng,
              n_monte_carlo_grad,
              1e4, // absurdly high!
              0.1,
              100,
              1,
              print_stream,
              &std::cout,
              &std::cout);

  // Create arbitrary variational family to calculate ELBO
  Eigen::VectorXd mu     = Eigen::VectorXd::Constant(my_model.num_params_r(),
                                                     1.88);
  Eigen::MatrixXd L_chol = Eigen::MatrixXd::Identity(my_model.num_params_r(),
                                                     my_model.num_params_r());
  // TODO this doesn't exist anymore
  stan::variational::normal_fullrank muL =
    stan::variational::normal_fullrank(mu, L_chol);

  double elbo = 0.0;
  elbo = test_advi.calc_ELBO(muL);

  // Calculate ELBO analytically
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
  elbo_true += -0.5 * ( pow(mu_j - 1.88, 2)/sigma_j_sq + 1.0/sigma_j_sq );
  elbo_true += 0.5 * (1 + log(2.0*stan::math::pi()));

  double const EPSILON = 0.1;
  EXPECT_NEAR(elbo_true, elbo, EPSILON);
}

TEST(AdviCalcElbo, univar_normal_meanfield) {
  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  Model_u my_model(dummy_context);
  rng_t base_rng(0);
  int n_monte_carlo_grad = 10;
  std::ostream* print_stream = &std::cout;
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(1);
  cont_params(0) = -0.75;

  stan::variational::advi<Model_u, stan::variational::normal_meanfield, rng_t>
    test_advi(my_model,
              cont_params,
              base_rng,
              n_monte_carlo_grad,
              1e4, // absurdly high!
              0.1,
              100,
              1,
              print_stream,
              &std::cout,
              &std::cout);

  // Create arbitrary variational family to calculate ELBO
  Eigen::VectorXd mu  = Eigen::VectorXd::Constant(my_model.num_params_r(),
                                                     1.88);
  Eigen::VectorXd sigma_tilde  = Eigen::VectorXd::Constant(
                                          my_model.num_params_r(),
                                          0.0); // initializing sigma_tilde = 0
                                                // means sigma = 1
  // TODO this doesn't exist anymore
  stan::variational::normal_meanfield musigmatilde =
    stan::variational::normal_meanfield(mu, sigma_tilde);

  double elbo = 0.0;
  elbo = test_advi.calc_ELBO(musigmatilde);

  // Calculate ELBO analytically
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
  elbo_true += -0.5 * ( pow(mu_j - 1.88, 2)/sigma_j_sq + 1.0/sigma_j_sq );
  elbo_true += 0.5 * (1 + log(2.0*stan::math::pi()));

  double const EPSILON = 0.1;
  EXPECT_NEAR(elbo_true, elbo, EPSILON);
}

TEST(AdviCalcElbo, univar_contraint_normal_fullrank) {
  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  Model_uc my_model(dummy_context);
  rng_t base_rng(0);
  int n_monte_carlo_grad = 10;
  std::ostream* print_stream = &std::cout;
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(1);
  cont_params(0) = -0.75;

  stan::variational::advi<Model_uc, stan::variational::normal_fullrank, rng_t>
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
  // TODO this doesn't exist anymore
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

TEST(AdviCalcElbo, univar_contraint_normal_meanfield) {
  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  Model_uc my_model(dummy_context);
  rng_t base_rng(0);
  int n_monte_carlo_grad = 10;
  std::ostream* print_stream = &std::cout;
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(1);
  cont_params(0) = -0.75;

  stan::variational::advi<Model_uc, stan::variational::normal_meanfield, rng_t>
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
  // TODO this doesn't exist anymore
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

TEST(AdviCalcElbo, multivar_normal_fullrank) {
  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  Model_m my_model(dummy_context);
  rng_t base_rng(0);
  int n_monte_carlo_grad = 10;
  int n_grad_samples = 1e4;
  std::ostream* print_stream = &std::cout;
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(2);
  cont_params(0) = 0.75;
  cont_params(1) = 0.75;

  stan::variational::advi<Model_m, stan::variational::normal_fullrank, rng_t>
    test_advi(my_model,
              cont_params,
              base_rng,
              n_monte_carlo_grad,
              n_grad_samples,
              0.1,
              100,
              1,
              print_stream,
              &std::cout,
              &std::cout);

  // Create arbitrary variational family to calculate ELBO
  Eigen::VectorXd mu     = Eigen::VectorXd::Constant(my_model.num_params_r(),
                                                      2.5);
  Eigen::MatrixXd L_chol = Eigen::MatrixXd::Identity(my_model.num_params_r(),
                                                     my_model.num_params_r());
  stan::variational::normal_fullrank muL =
    stan::variational::normal_fullrank(mu, L_chol);

  double elbo = 0.0;
  elbo = test_advi.calc_ELBO(muL);

  // Calculate ELBO analytically
  double zeta = -0.5 * ( 3*2*log(2.0*stan::math::pi()) + 18.5 + 25 + 13 );
  Eigen::VectorXd mu_J = Eigen::VectorXd::Zero(2);
  mu_J(0) = 10.5;
  mu_J(1) =  7.5;

  double elbo_true = 0.0;
  elbo_true += zeta;
  elbo_true += mu_J.dot(mu);
  elbo_true += -0.5 * ( 3*mu.dot(mu) + 3*2 );
  elbo_true += 1 + log(2.0*stan::math::pi());

  double const EPSILON = 0.1;
  EXPECT_NEAR(elbo_true, elbo, EPSILON);
}

TEST(AdviCalcElbo, multivar_normal_meanfield) {
  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  Model_m my_model(dummy_context);
  rng_t base_rng(0);
  int n_monte_carlo_grad = 10;
  std::ostream* print_stream = &std::cout;
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(2);
  cont_params(0) = 0.75;
  cont_params(1) = 0.75;

  stan::variational::advi<Model_m, stan::variational::normal_meanfield, rng_t>
    test_advi(my_model,
              cont_params,
              base_rng,
              n_monte_carlo_grad,
              1e4, // absurdly high!
              0.1,
              100,
              1,
              print_stream,
              &std::cout,
              &std::cout);

  // Create arbitrary variational family to calculate ELBO
  Eigen::VectorXd mu  = Eigen::VectorXd::Constant(my_model.num_params_r(),
                                                     2.5);
  Eigen::VectorXd sigma_tilde  = Eigen::VectorXd::Constant(
                                          my_model.num_params_r(),
                                          0.0); // initializing sigma_tilde = 0
                                                // means sigma = 1
  // TODO this doesn't exist anymore
  stan::variational::normal_meanfield musigmatilde =
    stan::variational::normal_meanfield(mu, sigma_tilde);

  double elbo = 0.0;
  elbo = test_advi.calc_ELBO(musigmatilde);

  // Calculate ELBO analytically
  double zeta = -0.5 * ( 3*2*log(2.0*stan::math::pi()) + 18.5 + 25 + 13 );
  Eigen::VectorXd mu_J = Eigen::VectorXd::Zero(2);
  mu_J(0) = 10.5;
  mu_J(1) =  7.5;

  double elbo_true = 0.0;
  elbo_true += zeta;
  elbo_true += mu_J.dot(mu);
  elbo_true += -0.5 * ( 3*mu.dot(mu) + 3*2 );
  elbo_true += 1 + log(2.0*stan::math::pi());

  double const EPSILON = 0.1;
  EXPECT_NEAR(elbo_true, elbo, EPSILON);
}

TEST(AdviCalcElbo, multivar_constraint_normal_fullrank) {
  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  Model_mc my_model(dummy_context);
  rng_t base_rng(0);
  int n_monte_carlo_grad = 10;
  int n_monte_carlo_elbo = 1e6;
  std::ostream* print_stream = &std::cout;
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(2);
  cont_params(0) = 0.75;
  cont_params(1) = 0.75;

  stan::variational::advi<Model_mc, stan::variational::normal_fullrank, rng_t>
    test_advi(my_model,
              cont_params,
              base_rng,
              n_monte_carlo_grad,
              n_monte_carlo_elbo,
              0.1,
              100,
              1,
              print_stream,
              &std::cout,
              &std::cout);

  // Create arbitrary variational family to calculate ELBO
  Eigen::VectorXd mu     = Eigen::VectorXd::Constant(my_model.num_params_r(),
                                                      log(2.5));
  Eigen::MatrixXd L_chol = Eigen::MatrixXd::Identity(my_model.num_params_r(),
                                                     my_model.num_params_r());
  // TODO this doesn't exist anymore
  stan::variational::normal_fullrank muL =
    stan::variational::normal_fullrank(mu, L_chol);

  double elbo = 0.0;
  elbo = test_advi.calc_ELBO(muL);

  // Calculate ELBO analytically
  double zeta = -0.5 * ( 3*2*log(2.0*stan::math::pi()) + 18.5 + 25 + 13 );
  Eigen::VectorXd mu_J = Eigen::VectorXd::Zero(2);
  mu_J(0) = 10.5;
  mu_J(1) =  7.5;

  double elbo_true = 0.0;
  elbo_true += zeta;
  elbo_true += 74.192457181505773; // mu_J.dot( (mu + 0.5).exp() );
  elbo_true += -0.5 * 3 * ( 92.363201236633131 );
  elbo_true += 2*log(2.5);
  elbo_true += 1 + log(2.0*stan::math::pi());

  double const EPSILON = 1.0;
  EXPECT_NEAR(elbo_true, elbo, EPSILON);
}

TEST(AdviCalcElbo, multivar_constraint_normal_meanfield) {
  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  Model_mc my_model(dummy_context);
  rng_t base_rng(0);
  int n_monte_carlo_grad = 10;
  int n_monte_carlo_elbo = 1e6;
  std::ostream* print_stream = &std::cout;
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(2);
  cont_params(0) = 0.75;
  cont_params(1) = 0.75;

  stan::variational::advi<Model_mc, stan::variational::normal_meanfield, rng_t>
    test_advi(my_model,
              cont_params,
              base_rng,
              n_monte_carlo_grad,
              n_monte_carlo_elbo,
              0.1,
              100,
              1,
              print_stream,
              &std::cout,
              &std::cout);

  // Create arbitrary variational family to calculate ELBO
  Eigen::VectorXd mu  = Eigen::VectorXd::Constant(my_model.num_params_r(),
                                                     log(2.5));
  Eigen::VectorXd sigma_tilde  = Eigen::VectorXd::Constant(
                                          my_model.num_params_r(),
                                          0.0); // initializing sigma_tilde = 0
                                                // means sigma = 1
  // TODO this doesn't exist anymore
  stan::variational::normal_meanfield musigmatilde =
    stan::variational::normal_meanfield(mu, sigma_tilde);

  double elbo = 0.0;
  elbo = test_advi.calc_ELBO(musigmatilde);

  // Calculate ELBO analytically
  double zeta = -0.5 * ( 3*2*log(2.0*stan::math::pi()) + 18.5 + 25 + 13 );
  Eigen::VectorXd mu_J = Eigen::VectorXd::Zero(2);
  mu_J(0) = 10.5;
  mu_J(1) =  7.5;

  double elbo_true = 0.0;
  elbo_true += zeta;
  elbo_true += 74.192457181505773; //;mu_J.dot( (mu + 0.5).exp() );
  elbo_true += -0.5 * 3 * ( 92.363201236633131 );
  elbo_true += 2*log(2.5);
  elbo_true += 1 + log(2.0*stan::math::pi());

  double const EPSILON = 1.0;
  EXPECT_NEAR(elbo_true, elbo, EPSILON);
}
#endif
