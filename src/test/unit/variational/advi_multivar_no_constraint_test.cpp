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
  std::cout << elbo << std::endl;

  double zeta = -0.5 * ( 3*2*log(2.0*stan::math::pi()) + 18.5 + 25 + 13 );
  Eigen::VectorXd mu_J = Eigen::VectorXd::Zero(2);
  mu_J(0) = 10.5;
  mu_J(1) =  7.5;

  double elbo_true = 0.0;
  elbo_true += zeta;
  elbo_true += mu_J.dot(mu);
  elbo_true += -0.5 * ( mu.dot(mu) + 3*2 );
  elbo_true += 1 + log(2.0*stan::math::pi());
  std::cout << elbo_true << std::endl;

  // Can calculate ELBO using formula for E_normal[(something - x)^2]
  // (see Sam Roweis' notes on Gaussian identities)
  //
  // mu0 = np.array([3.7, 2.7])
  // x1  = np.array([4.0, 3.0])
  // x2  = np.array([3.5, 2.5])
  // z   = np.array([5, 5])
  //
  // ELBO = - 0.5 * ( 6 + np.linalg.norm(x1-z)**2
  //                    + np.linalg.norm(x2-z)**2
  //                    + np.linalg.norm(z-mu0)**2 ) + 2
  //

  // FIXME: perhaps make EPSILON depend on stan::vb::bbvb.n_monte_carlo_ ?
  double const EPSILON = 1.0;
  EXPECT_NEAR(elbo_true, elbo, EPSILON);

}



