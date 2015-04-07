#include <test/test-models/good/variational/multivariate_no_constraint.hpp>
#include <stan/variational/advi.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG

typedef boost::ecuyer1988 rng_t;
typedef multivariate_no_constraint_model_namespace::multivariate_no_constraint_model Model;

TEST(advi_test, multivar_no_constraint_fullrank_ELBO) {

  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  // Instantiate model
  Model my_model(dummy_context);

  // RNG
  rng_t base_rng(0);

  // Dummy input
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(2);
  cont_params(0) = 0.75;
  cont_params(1) = 0.75;

  // ADVI
  stan::variational::advi<Model, rng_t> test_advi(my_model,
                                                  cont_params,
                                                  0,
                                                  10,
                                                  1e4, // absurdly high!
                                                  0.1,
                                                  base_rng,
                                                  100,
                                                  &std::cout,
                                                  &std::cout,
                                                  &std::cout);

  // Create some arbitrary variational q() family to calculate the ELBO over
  Eigen::VectorXd mu     = Eigen::VectorXd::Constant(my_model.num_params_r(),
                                                      2.5);
  Eigen::MatrixXd L_chol = Eigen::MatrixXd::Identity(my_model.num_params_r(),
                                                     my_model.num_params_r());
  stan::variational::advi_params_fullrank muL =
    stan::variational::advi_params_fullrank(mu, L_chol);

  double elbo = 0.0;
  elbo = test_advi.calc_ELBO(muL, 0.0);//constant_factor);

  // Can calculate ELBO analytically
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



