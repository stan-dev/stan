#include <stan/variational/families/normal_fullrank.hpp>
#include <vector>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(normal_fullrank_test, zero_init) {
  int my_dimension =  10;

  stan::variational::normal_fullrank my_normal_fullrank(my_dimension);
  EXPECT_FLOAT_EQ(my_dimension, my_normal_fullrank.dimension());

  const Eigen::VectorXd& mu_out     = my_normal_fullrank.mu();
  const Eigen::MatrixXd& L_chol_out = my_normal_fullrank.L_chol();

  for (int i = 0; i < my_dimension; ++i) {
    EXPECT_FLOAT_EQ(0.0, mu_out(i));
    for (int j = 0; j < my_dimension; ++j) {
      EXPECT_FLOAT_EQ(0.0, L_chol_out(i,j));
    }
  }
}

TEST(normal_fullrank_test, dimension) {
  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0, 0,
       2.3, 41, 0,
       3.3, 42, 92;

  stan::variational::normal_fullrank my_normal_fullrank(mu, L);

  EXPECT_FLOAT_EQ(mu.size(), my_normal_fullrank.dimension());
}

TEST(normal_fullrank_test, mean_vector) {
  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0, 0,
       2.3, 41, 0,
       3.3, 42, 92;

  stan::variational::normal_fullrank my_normal_fullrank(mu, L);

  const Eigen::Vector3d& mu_out = my_normal_fullrank.mu();

  for (int i = 0; i < my_normal_fullrank.dimension(); ++i)
    EXPECT_FLOAT_EQ(mu(i), mu_out(i));


  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::Vector3d mu_nan = Eigen::VectorXd::Constant(3, nan);

  EXPECT_THROW(stan::variational::normal_fullrank my_normal_fullrank_nan(mu_nan, L);,
                   std::domain_error);
  EXPECT_THROW(my_normal_fullrank.set_mu(mu_nan);,
                   std::domain_error);
  Eigen::MatrixXd L_nan = Eigen::MatrixXd::Constant(3,3,nan);
  EXPECT_THROW(stan::variational::normal_fullrank my_normal_fullrank_nan(mu, L_nan);,
                   std::domain_error);

  my_normal_fullrank.set_to_zero();
  const Eigen::Vector3d& mu_out_zero = my_normal_fullrank.mu();

  for (int i = 0; i < my_normal_fullrank.dimension(); ++i) {
    EXPECT_FLOAT_EQ(0.0, mu_out_zero(i));
  }
}

TEST(normal_fullrank_test, cholesky_factor) {
  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0, 0,
       2.3, 41, 0,
       3.3, 42, 92;

  stan::variational::normal_fullrank my_normal_fullrank(mu, L);

  const Eigen::Matrix3d& L_out = my_normal_fullrank.L_chol();

  for (int j = 0, nRows = L.rows(), nCols = L.cols(); j < nCols; ++j) {
    for (int i = 0; i < nRows; ++i) {
      EXPECT_FLOAT_EQ(L(i, j), L_out(i, j));
    }
  }

  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd L_nan = Eigen::MatrixXd::Constant(3,3,nan);
  EXPECT_THROW(my_normal_fullrank.set_L_chol(L_nan),
                   std::domain_error);

  my_normal_fullrank.set_to_zero();
  const Eigen::Matrix3d& L_out_zero = my_normal_fullrank.L_chol();

  for (int i = 0; i < my_normal_fullrank.dimension(); ++i) {
    for (int j = 0; j < my_normal_fullrank.dimension(); ++j) {
      EXPECT_FLOAT_EQ(0.0, L_out_zero(i,j));
    }
  }

}

TEST(normal_fullrank_test, entropy) {
  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0,  0,
       2.3, 41, 0,
       3.3, 42, 92;

  stan::variational::normal_fullrank my_normal_fullrank(mu, L);

  double entropy_true = 12.754540507834857;

  const double entropy_out = my_normal_fullrank.entropy();

  EXPECT_FLOAT_EQ(entropy_out, entropy_true);
}

TEST(normal_fullrank_test, transform) {
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

  stan::variational::normal_fullrank my_normal_fullrank(mu, L);

  Eigen::Vector3d x_result;
  x_result = my_normal_fullrank.transform(x);

  for (int i = 0; i < my_normal_fullrank.dimension(); ++i)
    EXPECT_FLOAT_EQ(x_result(i), x_transformed(i));

  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::Vector3d x_nan = Eigen::VectorXd::Constant(3, nan);
  EXPECT_THROW(my_normal_fullrank.transform(x_nan);,
                   std::domain_error);
}

TEST(normal_fullrank_test, calc_log_g) {
  Eigen::Vector3d x;
  x << 7.1, -9.2, 0.59;
  
  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::Matrix3d L;
  L << 1.3, 0,  0,
       2.3, 41, 0,
       3.3, 42, 92;

  stan::variational::normal_fullrank my_normal_fullrank(mu, L);

  double log_g_true = -67.699049999999985;

  const double log_g_out = my_normal_fullrank.calc_log_g(x);

  EXPECT_FLOAT_EQ(log_g_out, log_g_true);
}

