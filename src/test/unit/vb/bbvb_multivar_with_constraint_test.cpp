#include <stan/vb/bbvb.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <string>

#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG

#include <test/test-models/no-main/vb/multivariate_with_constraint.cpp>

typedef boost::ecuyer1988 rng_t;
typedef multivariate_with_constraint_model_namespace::multivariate_with_constraint_model Model;

TEST(bbvb_test, multivar_with_constraint_ELBO) {

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

  // VB
  stan::vb::bbvb<Model, rng_t> test_vb(my_model, cont_params, base_rng, &std::cout, &std::cout);

  // Create some arbitrary variational q() family to calculate the ELBO over
  Eigen::VectorXd mu = Eigen::VectorXd::Constant(my_model.num_params_r(), 5.0);
  Eigen::MatrixXd L  = Eigen::MatrixXd::Identity(my_model.num_params_r(), my_model.num_params_r());

  // latent_vars will store the parametrization of q()
  stan::vb::latent_vars muL = stan::vb::latent_vars(mu,L);

  double elbo = 0.0;
  elbo = test_vb.calc_ELBO(muL);

  // Can calculate ELBO using moment generating function of Gaussians
  //
  // mu0 = np.array([3.7, 2.7])
  // x1  = np.array([4.0, 3.0])
  // x2  = np.array([3.5, 2.5])
  //
  // ELBO = -0.5 * ( np.exp(5*2+2) - 2*x1[0]*np.exp(5+0.5) + (x1[0])**2 +
  //                 np.exp(5*2+2) - 2*x1[1]*np.exp(5+0.5) + (x1[1])**2 +
  //                 np.exp(5*2+2) - 2*x2[0]*np.exp(5+0.5) + (x2[0])**2 +
  //                 np.exp(5*2+2) - 2*x2[1]*np.exp(5+0.5) + (x2[1])**2 +
  //                 np.exp(5*2+2) - 2*mu0[0]*np.exp(5+0.5) + (mu0[0])**2 +
  //                 np.exp(5*2+2) - 2*mu0[1]*np.exp(5+0.5) + (mu0[1])**2 ) +
  //                 10.0
  //

  // FIXME: perhaps make EPSILON depend on stan::vb::bbvb.n_monte_carlo_ ?
  double const EPSILON = 10000.0;
  EXPECT_NEAR(-483539.590, elbo, EPSILON);

}
