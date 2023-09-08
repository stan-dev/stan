#include <string>
#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

TEST(McmcDiagEPoint, inv_metric_wrong_size) {
  int N = 2;

  stan::mcmc::diag_e_point z(N);

  Eigen::VectorXd inv_metric_large(2 * N);

  EXPECT_THROW_MSG(z.set_inv_metric(inv_metric_large), std::invalid_argument,
                   "number of rows in new inverse metric");
}

TEST(McmcDiagEPoint, inv_metric) {
  int N = 2;

  stan::mcmc::diag_e_point z(N);

  Eigen::VectorXd inv_metric(N);

  inv_metric << 2, 2;

  z.set_inv_metric(inv_metric);

  Eigen::VectorXd z_inv_metric = z.get_inv_metric();
  EXPECT_EQ(z_inv_metric.size(), inv_metric.size());
  for (size_t i = 0; i < inv_metric.size(); ++i)
    EXPECT_FLOAT_EQ(z_inv_metric(i), inv_metric(i));
}
