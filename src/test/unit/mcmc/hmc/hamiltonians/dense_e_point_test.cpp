#include <string>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

TEST(McmcDenseEPoint, inv_metric_wrong_size) {
  int N = 2;

  stan::mcmc::dense_e_point z(N);

  Eigen::MatrixXd inv_metric_large(2 * N, 2 * N);

  EXPECT_THROW_MSG(z.set_inv_metric(inv_metric_large), std::invalid_argument,
                   "number of rows in new inverse metric");
}

TEST(McmcDenseEPoint, inv_metric) {
  int N = 2;

  stan::mcmc::dense_e_point z(N);

  Eigen::MatrixXd inv_metric(N, N);

  inv_metric << 2, 1, 1, 2;

  z.set_inv_metric(inv_metric);

  Eigen::MatrixXd z_inv_metric = z.get_inv_metric();
  EXPECT_EQ(z_inv_metric.rows(), inv_metric.rows());
  EXPECT_EQ(z_inv_metric.cols(), inv_metric.cols());
  for (size_t i = 0; i < inv_metric.size(); ++i)
    EXPECT_FLOAT_EQ(z_inv_metric(i), inv_metric(i));
}

TEST(McmcDenseEPoint, inv_metric_llt) {
  int N = 2;

  stan::mcmc::dense_e_point z(N);

  Eigen::MatrixXd inv_metric(N, N);

  inv_metric << 2, 1, 1, 2;

  z.set_inv_metric(inv_metric);

  Eigen::MatrixXd z_inv_metric_llt_matrixU = z.get_transpose_llt_inv_metric();
  Eigen::MatrixXd z_inv_metric_2
      = z_inv_metric_llt_matrixU.transpose() * z_inv_metric_llt_matrixU;
  EXPECT_EQ(z_inv_metric_2.rows(), inv_metric.rows());
  EXPECT_EQ(z_inv_metric_2.cols(), inv_metric.cols());
  for (size_t i = 0; i < inv_metric.size(); ++i)
    EXPECT_FLOAT_EQ(z_inv_metric_2(i), inv_metric(i));
}
