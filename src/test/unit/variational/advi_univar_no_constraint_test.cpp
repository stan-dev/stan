#include <stan/variational/advi.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <string>

#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG

#include <test/test-models/good/variational/univariate_no_constraint.hpp>

typedef boost::ecuyer1988 rng_t;
typedef univariate_no_constraint_model_namespace::univariate_no_constraint_model Model;

TEST(advi_test, univar_no_constraint_fullrank_ELBO) {

  // Create mock data_var_context
  static const std::string DATA = "";
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);

  // Instantiate model
  Model my_model(dummy_context);

  // RNG
  rng_t base_rng(0);

  // Dummy input
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(1);
  cont_params(0) = -0.75;

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
  std::cout << elbo << std::endl;

  // Can calculate ELBO analytically
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
  elbo_true += -0.5 * ( pow(mu_j - 2.5, 2)/sigma_j_sq + 1.0/sigma_j_sq );
  elbo_true += 0.5 * (1 + log(2.0*stan::math::pi()));

  std::cout << sigma_j_sq << std::endl;
  std::cout << mu_j << std::endl;
  std::cout << S_j << std::endl;
  std::cout << elbo_true << std::endl;

  double const EPSILON = 0.1;
  EXPECT_NEAR(elbo_true, elbo, EPSILON);

}



