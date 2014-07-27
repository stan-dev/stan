#include <stan/vb/latent_vars.hpp>
#include <vector>
#include <gtest/gtest.h>

TEST(latent_vars_test, dimension) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0, 0,
       2.3, 41, 0,
       3.3, 42, 92;

  stan::vb::latent_vars my_latent_vars(mu, L);

  EXPECT_EQ(mu.size(), my_latent_vars.dimension());

}

TEST(latent_vars_test, mean_vector) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0, 0,
       2.3, 41, 0,
       3.3, 42, 92;

  stan::vb::latent_vars my_latent_vars(mu, L);

  const Eigen::Vector3d& mu_out = my_latent_vars.mu();

  for (int i = 0; i < my_latent_vars.dimension(); ++i)
    EXPECT_EQ(mu(i), mu_out(i));

}

TEST(latent_vars_test, cholesky_factor) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0, 0,
       2.3, 41, 0,
       3.3, 42, 92;

  stan::vb::latent_vars my_latent_vars(mu, L);

  const Eigen::Matrix3d& L_out = my_latent_vars.L();

  for (int j = 0, nRows = L.rows(), nCols = L.cols(); j < nCols; ++j) {
    for (int i = 0; i < nRows; ++i) {
      EXPECT_EQ(L(i, j), L_out(i, j));
    }
  }

}

TEST(latent_vars_test, transform_to_unconstrained) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0, 0,
       2.3, 41, 0,
       3.3, 42, 92;

  Eigen::VectorXd x(3);
  x << 7.1, -9.2, 0.59;

  Eigen::Vector3d x_transformed;
  x_transformed << 14.93, -364.07, -308.5568;

  stan::vb::latent_vars my_latent_vars(mu, L);

  my_latent_vars.to_unconstrained(x);

  for (int i = 0; i < my_latent_vars.dimension(); ++i)
    EXPECT_DOUBLE_EQ(x(i), x_transformed(i));

}

TEST(latent_vars_test, transform_to_standardized) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0, 0,
       2.3, 41, 0,
       3.3, 42, 92;

  Eigen::VectorXd x(3);
  x << 14.93, -364.07, -308.5568;

  Eigen::Vector3d x_standardized;
  x_standardized << 7.1, -9.2, 0.59;

  stan::vb::latent_vars my_latent_vars(mu, L);

  my_latent_vars.to_standardized(x);

  for (int i = 0; i < my_latent_vars.dimension(); ++i)
    EXPECT_DOUBLE_EQ(x(i), x_standardized(i));

}
