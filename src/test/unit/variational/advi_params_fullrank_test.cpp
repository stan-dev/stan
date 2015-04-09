#include <stan/variational/advi_params_fullrank.hpp>
#include <vector>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(advi_params_fullrank_test, dimension) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0, 0,
       2.3, 41, 0,
       3.3, 42, 92;

  stan::variational::advi_params_fullrank my_advi_params_fullrank(mu, L);

  EXPECT_FLOAT_EQ(mu.size(), my_advi_params_fullrank.dimension());

}

TEST(advi_params_fullrank_test, mean_vector) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0, 0,
       2.3, 41, 0,
       3.3, 42, 92;

  stan::variational::advi_params_fullrank my_advi_params_fullrank(mu, L);

  const Eigen::Vector3d& mu_out = my_advi_params_fullrank.mu();

  for (int i = 0; i < my_advi_params_fullrank.dimension(); ++i)
    EXPECT_FLOAT_EQ(mu(i), mu_out(i));


  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::Vector3d mu_nan = Eigen::VectorXd::Constant(3, nan);

  std::string error = "stan::variational::advi_params_fullrank: "
                      "Mean vector is nan, but must not be nan!";
  EXPECT_THROW_MSG(stan::variational::advi_params_fullrank my_advi_params_fullrank_nan(mu_nan, L);,
                   std::domain_error, error);

  error = "stan::variational::advi_params_fullrank::set_mu: "
          "Input vector is nan, but must not be nan!";
  EXPECT_THROW_MSG(my_advi_params_fullrank.set_mu(mu_nan);,
                   std::domain_error, error);

  Eigen::MatrixXd L_nan = Eigen::MatrixXd::Constant(3,3,nan);
  error = "stan::variational::advi_params_fullrank: "
          "Cholesky factor is not lower triangular; Cholesky factor[1,2]=nan";
  EXPECT_THROW_MSG(stan::variational::advi_params_fullrank my_advi_params_fullrank_nan(mu, L_nan);,
                   std::domain_error, error);

}

TEST(advi_params_fullrank_test, cholesky_factor) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0, 0,
       2.3, 41, 0,
       3.3, 42, 92;

  stan::variational::advi_params_fullrank my_advi_params_fullrank(mu, L);

  const Eigen::Matrix3d& L_out = my_advi_params_fullrank.L_chol();

  for (int j = 0, nRows = L.rows(), nCols = L.cols(); j < nCols; ++j) {
    for (int i = 0; i < nRows; ++i) {
      EXPECT_FLOAT_EQ(L(i, j), L_out(i, j));
    }
  }

}

TEST(advi_params_fullrank_test, entropy) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0,  0,
       2.3, 41, 0,
       3.3, 42, 92;

  stan::variational::advi_params_fullrank my_advi_params_fullrank(mu, L);

  double entropy_true = 12.754540507834857;

  const double entropy_out = my_advi_params_fullrank.entropy();

  EXPECT_FLOAT_EQ(entropy_out, entropy_true);

}

TEST(advi_params_fullrank_test, transform_to_unconstrained) {

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0, 0,
       2.3, 41, 0,
       3.3, 42, 92;

  Eigen::Vector3d x;
  x << 7.1, -9.2, 0.59;

  Eigen::Vector3d x_transformed;
  x_transformed << 14.93, -364.07, -308.5568;

  stan::variational::advi_params_fullrank my_advi_params_fullrank(mu, L);

  Eigen::Vector3d x_result;
  x_result = my_advi_params_fullrank.to_unconstrained(x);

  for (int i = 0; i < my_advi_params_fullrank.dimension(); ++i)
    EXPECT_FLOAT_EQ(x_result(i), x_transformed(i));

  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::Vector3d x_nan = Eigen::VectorXd::Constant(3, nan);

  std::string error = "stan::variational::advi_params_fullrank::to_unconstrained: "
          "Input vector is nan, but must not be nan!";
  EXPECT_THROW_MSG(my_advi_params_fullrank.to_unconstrained(x_nan);,
                   std::domain_error, error);
}
