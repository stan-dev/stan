#include <stan/variational/advi_params_meanfield.hpp>
#include <vector>
#include <gtest/gtest.h>

TEST(advi_params_meanfield_test, dimension) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Vector3d sigma_tilde;
  sigma_tilde << -0.42, 0.8922, 13.4;

  stan::variational::advi_params_meanfield my_advi_params_meanfield(mu, sigma_tilde);

  EXPECT_FLOAT_EQ(mu.size(), my_advi_params_meanfield.dimension());
  EXPECT_FLOAT_EQ(sigma_tilde.size(), my_advi_params_meanfield.dimension());

}

TEST(advi_params_meanfield_test, mean_vector) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Vector3d sigma_tilde;
  sigma_tilde << -0.42, 0.8922, 13.4;

  stan::variational::advi_params_meanfield my_advi_params_meanfield(mu, sigma_tilde);

  const Eigen::Vector3d& mu_out = my_advi_params_meanfield.mu();

  for (int i = 0; i < my_advi_params_meanfield.dimension(); ++i)
    EXPECT_FLOAT_EQ(mu(i), mu_out(i));

}

TEST(advi_params_meanfield_test, sigma_tilde_vector) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Vector3d sigma_tilde;
  sigma_tilde << -0.42, 0.8922, 13.4;

  stan::variational::advi_params_meanfield my_advi_params_meanfield(mu, sigma_tilde);

  const Eigen::Vector3d& sigma_tilde_out = my_advi_params_meanfield.sigma_tilde();

  for (int i = 0; i < my_advi_params_meanfield.dimension(); ++i)
    EXPECT_FLOAT_EQ(sigma_tilde(i), sigma_tilde_out(i));

}

TEST(advi_params_meanfield_test, entropy) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Vector3d sigma_tilde;
  sigma_tilde << -0.42, 0.8922, 13.4;

  double entropy_true = 18.129015599614018;

  stan::variational::advi_params_meanfield my_advi_params_meanfield(mu, sigma_tilde);

  const double entropy_out = my_advi_params_meanfield.entropy();

  EXPECT_FLOAT_EQ(entropy_out, entropy_true);

}

TEST(advi_params_meanfield_test, transform_to_unconstrained) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Vector3d sigma_tilde;
  sigma_tilde << -0.42, 0.8922, 13.4;

  stan::variational::advi_params_meanfield my_advi_params_meanfield(mu, sigma_tilde);

  Eigen::Vector3d x;
  x << 7.1, -9.2, 0.59;

  Eigen::Vector3d x_transformed;
  x_transformed << 1.036503242068690e01,  -2.565253407151558e01,
         3.894020358120325e05;

  Eigen::Vector3d x_result;
  x_result = my_advi_params_meanfield.to_unconstrained(x);

  for (int i = 0; i < my_advi_params_meanfield.dimension(); ++i)
    EXPECT_FLOAT_EQ(x_result(i), x_transformed(i));

}
