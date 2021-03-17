#include <stan/variational/families/normal_lowrank.hpp>
#include <vector>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(normal_lowrank_test, zero_init) {
  int my_dimension = 10;
  int my_rank = 3;

  stan::variational::normal_lowrank my_normal_lowrank(my_dimension, my_rank);
  EXPECT_FLOAT_EQ(my_dimension, my_normal_lowrank.dimension());
  EXPECT_FLOAT_EQ(my_rank, my_normal_lowrank.rank());

  const Eigen::VectorXd& mu_out = my_normal_lowrank.mu();
  const Eigen::MatrixXd& B_out = my_normal_lowrank.B();
  const Eigen::MatrixXd& log_d_out = my_normal_lowrank.log_d();

  for (int i = 0; i < my_dimension; ++i) {
    EXPECT_FLOAT_EQ(0.0, mu_out(i));
    EXPECT_FLOAT_EQ(0.0, log_d_out(i));
    for (int j = 0; j < my_rank; ++j) {
      EXPECT_FLOAT_EQ(0.0, B_out(i, j));
    }
  }
}

TEST(normal_lowrank_test, dimension_and_rank) {
  Eigen::VectorXd mu(4);
  mu << 5.7, -3.2, 0.1332, -1.87;

  Eigen::MatrixXd B(4, 3);
  B << 1.3, 0, 0, 2.3, 41, 0, 3.3, 42, 92, 4.3, 43, 93;

  Eigen::VectorXd d(4);
  d << 0.63, 0.94, 1.32, 1.18;
  Eigen::VectorXd log_d = d.array().log().matrix();

  stan::variational::normal_lowrank my_normal_lowrank(mu, B, d);

  EXPECT_FLOAT_EQ(mu.size(), my_normal_lowrank.dimension());
  EXPECT_FLOAT_EQ(B.cols(), my_normal_lowrank.rank());
}

TEST(normal_lowrank_test, mean_vector) {
  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::MatrixXd B(3, 2);
  B << 1.3, 0, 2.3, 41, 3.3, 42;

  Eigen::Vector3d d;
  d << 0.63, 0.94, 1.32;
  Eigen::VectorXd log_d = d.array().log().matrix();

  stan::variational::normal_lowrank my_normal_lowrank(mu, B, log_d);

  const Eigen::Vector3d& mu_out = my_normal_lowrank.mu();

  for (int i = 0; i < my_normal_lowrank.dimension(); ++i)
    EXPECT_FLOAT_EQ(mu(i), mu_out(i));

  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::Vector3d mu_nan = Eigen::VectorXd::Constant(3, nan);

  EXPECT_THROW(
      stan::variational::normal_lowrank my_normal_lowrank_nan(mu_nan, B, log_d);
      , std::domain_error);
  EXPECT_THROW(my_normal_lowrank.set_mu(mu_nan);, std::domain_error);
  Eigen::MatrixXd B_nan = Eigen::MatrixXd::Constant(3, 3, nan);
  EXPECT_THROW(
      stan::variational::normal_lowrank my_normal_lowrank_nan(mu, B_nan, log_d);
      , std::domain_error);

  my_normal_lowrank.set_to_zero();
  const Eigen::Vector3d& mu_out_zero = my_normal_lowrank.mu();

  for (int i = 0; i < my_normal_lowrank.dimension(); ++i) {
    EXPECT_FLOAT_EQ(0.0, mu_out_zero(i));
  }
}

TEST(normal_lowrank_test, lowrank_factor) {
  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::MatrixXd B(3, 2);
  B << 1.3, 0, 2.3, 41, 3.3, 42;

  Eigen::Vector3d d;
  d << 0.63, 0.94, 1.32;
  Eigen::VectorXd log_d = d.array().log().matrix();

  stan::variational::normal_lowrank my_normal_lowrank(mu, B, log_d);

  const Eigen::MatrixXd& B_out = my_normal_lowrank.B();

  for (int j = 0, nRows = B.rows(), nCols = B.cols(); j < nCols; ++j) {
    for (int i = 0; i < nRows; ++i) {
      EXPECT_FLOAT_EQ(B(i, j), B_out(i, j));
    }
  }

  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd B_nan = Eigen::MatrixXd::Constant(3, 2, nan);
  EXPECT_THROW(my_normal_lowrank.set_B(B_nan);, std::domain_error);
  my_normal_lowrank.set_to_zero();
  const Eigen::MatrixXd& B_out_zero = my_normal_lowrank.B();

  for (int i = 0; i < my_normal_lowrank.dimension(); ++i) {
    for (int j = 0; j < my_normal_lowrank.rank(); ++j) {
      EXPECT_FLOAT_EQ(0.0, B_out_zero(i, j));
    }
  }
}

TEST(normal_lowrank_test, entropy) {
  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::MatrixXd B(3, 2);
  B << 1.3, 0, 2.3, 41, 3.3, 42;

  Eigen::Vector3d d;
  d << 0.63, 0.94, 1.32;
  Eigen::VectorXd log_d = d.array().log().matrix();

  stan::variational::normal_lowrank my_normal_lowrank(mu, B, log_d);

  double entropy_true = 8.86050306582748;

  const double entropy_out = my_normal_lowrank.entropy();

  EXPECT_FLOAT_EQ(entropy_out, entropy_true);
}

TEST(normal_lowrank_test, transform) {
  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::MatrixXd B(3, 2);
  B << 1.3, 0, 2.3, 41, 3.3, 42;

  Eigen::Vector3d d;
  d << 0.63, 0.94, 0.1332;
  Eigen::VectorXd log_d = d.array().log().matrix();

  Eigen::VectorXd x(5);
  x << 7.1, -9.2, 0.59, 0.24, -1.92;

  Eigen::Vector3d x_transformed;
  x_transformed << 15.3017, -363.8444, -363.092544;

  stan::variational::normal_lowrank my_normal_lowrank(mu, B, log_d);

  Eigen::Vector3d x_result;
  x_result = my_normal_lowrank.transform(x);

  for (int i = 0; i < my_normal_lowrank.dimension(); ++i)
    EXPECT_FLOAT_EQ(x_result(i), x_transformed(i));

  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd x_nan = Eigen::VectorXd::Constant(5, nan);
  EXPECT_THROW(my_normal_lowrank.transform(x_nan);, std::domain_error);
}

TEST(normal_lowrank_test, calc_log_g) {
  Eigen::VectorXd x(5);
  x << 7.1, -9.2, 0.59, 0.24, -1.92;

  Eigen::Vector3d mu;
  mu << 5.7, -3.2, 0.1332;

  Eigen::MatrixXd B(3, 2);
  B << 1.3, 0, 2.3, 41, 3.3, 42;

  Eigen::Vector3d d;
  d << 0.63, 0.94, 0.1332;
  Eigen::VectorXd log_d = d.array().log().matrix();

  stan::variational::normal_lowrank my_normal_lowrank(mu, B, log_d);

  double log_g_true = -69.57104999999999;

  const double log_g_out = my_normal_lowrank.calc_log_g(x);

  EXPECT_FLOAT_EQ(log_g_out, log_g_true);
}
