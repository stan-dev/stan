#include <stan/mcmc/auto_adaptation.hpp>
#include <test/test-models/good/model/known_hessian.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <gtest/gtest.h>

TEST(McmcAutoAdaptation, test_covariance_zero_rows_zero_cols) {
  Eigen::MatrixXd X1(0, 5);

  EXPECT_THROW(stan::mcmc::internal::covariance(X1), std::invalid_argument);

  Eigen::MatrixXd X2(1, 0);

  EXPECT_THROW(stan::mcmc::internal::covariance(X2), std::invalid_argument);
}

TEST(McmcAutoAdaptation, test_covariance_one_row_one_col) {
  Eigen::MatrixXd X1(1, 2);
  Eigen::MatrixXd X2(3, 1);

  X1 << 1.0, 2.0;
  X2 << 1.0, 2.0, 3.0;

  Eigen::MatrixXd cov1 = stan::mcmc::internal::covariance(X1);
  Eigen::MatrixXd cov2 = stan::mcmc::internal::covariance(X2);

  ASSERT_EQ(cov1.rows(), 2);
  ASSERT_EQ(cov1.cols(), 2);

  ASSERT_EQ(cov2.rows(), 1);
  ASSERT_EQ(cov2.cols(), 1);

  for (int i = 0; i < cov1.size(); ++i) {
    ASSERT_FLOAT_EQ(cov1(i), 0.0);
  }

  ASSERT_FLOAT_EQ(cov2(0), 1.0);
}

TEST(McmcAutoAdaptation, test_covariance) {
  Eigen::MatrixXd X1(3, 2);
  Eigen::MatrixXd X2(2, 3);

  X1 << 0.0, -1.0, 0.5, -2.7, 3.0, 5.0;
  X2 << 0.0, 3, -2.7, 0.5, -1, 5.0;

  Eigen::MatrixXd cov1 = stan::mcmc::internal::covariance(X1);
  Eigen::MatrixXd cov2 = stan::mcmc::internal::covariance(X2);

  Eigen::MatrixXd cov1_ref(2, 2);
  Eigen::MatrixXd cov2_ref(3, 3);

  cov1_ref << 2.5833333333333335, 6.0666666666666664, 6.0666666666666664,
      16.3633333333333333;

  cov2_ref << 0.125, -1.0, 1.925, -1.000, 8.0, -15.4, 1.925, -15.4, 29.645;

  ASSERT_EQ(cov1.rows(), cov1_ref.rows());
  ASSERT_EQ(cov1.cols(), cov1_ref.cols());

  ASSERT_EQ(cov2.rows(), cov2_ref.rows());
  ASSERT_EQ(cov2.cols(), cov2_ref.cols());

  for (int i = 0; i < cov1_ref.size(); ++i) {
    ASSERT_FLOAT_EQ(cov1(i), cov1_ref(i));
  }

  for (int i = 0; i < cov2_ref.size(); ++i) {
    ASSERT_FLOAT_EQ(cov2(i), cov2_ref(i));
  }
}

TEST(McmcAutoAdaptation, power_method) {
  Eigen::MatrixXd X(2, 2);
  Eigen::VectorXd x0(2);

  X << 2.0, 0.5, 0.5, 1.0;
  x0 << 1.0, 0.0;

  const int max_iterations = 10;
  const double tol = 1e-10;

  auto Av = [&](const Eigen::VectorXd& v) { return X * v; };

  int max_iterations_1 = max_iterations;
  double tol_1 = tol;

  double eval
      = stan::mcmc::internal::power_method(Av, x0, max_iterations_1, tol_1);

  EXPECT_FLOAT_EQ(eval, 2.20710678118654746);
}

TEST(McmcAutoAdaptation, power_method_tol_check) {
  Eigen::MatrixXd X(2, 2);
  Eigen::VectorXd x0(2);

  X << 2.0, 0.5, 0.5, 1.0;
  x0 << 1.0, 0.0;

  const int max_iterations = 1000;
  const double tol = 1e-12;

  auto Av = [&](const Eigen::VectorXd& v) { return X * v; };

  int max_iterations_1 = max_iterations;
  double tol_1 = tol;
  double eval
      = stan::mcmc::internal::power_method(Av, x0, max_iterations_1, tol_1);

  EXPECT_LT(tol_1, tol);
}

TEST(McmcAutoAdaptation, power_method_iter_check) {
  Eigen::MatrixXd X(2, 2);
  Eigen::VectorXd x0(2);

  X << 2.0, 0.5, 0.5, 1.0;
  x0 << 1.0, 0.0;

  const int max_iterations = 10;
  const double tol = 1e-50;

  auto Av = [&](const Eigen::VectorXd& v) { return X * v; };

  int max_iterations_1 = max_iterations;
  double tol_1 = tol;
  double eval
      = stan::mcmc::internal::power_method(Av, x0, max_iterations_1, tol_1);

  EXPECT_GT(tol_1, tol);
  EXPECT_EQ(max_iterations_1, max_iterations);
}

// The checks in here are very coarse because eigenvalue_scaled_covariance
// only estimates things with low precision
TEST(McmcAutoAdaptation, eigenvalue_scaled_covariance) {
  Eigen::MatrixXd L(2, 2), Sigma(2, 2);

  L << 1.0, 0.0, 0.5, 1.0;
  Sigma << 2.0, 0.7, 0.7, 1.3;

  double eval = stan::mcmc::internal::eigenvalue_scaled_covariance(L, Sigma);

  EXPECT_LT(std::abs(eval - 2.0908326913195983) / eval, 1e-2);

  L << 2.0, 0.0, 0.7, 1.3;

  eval = stan::mcmc::internal::eigenvalue_scaled_covariance(L, Sigma);

  EXPECT_LT(std::abs(eval - 0.62426035502958577) / eval, 1e-2);
}

// The checks in here are very coarse because eigenvalue_scaled_hessian
// only estimates things with low precision
TEST(McmcAutoAdaptation, eigenvalue_scaled_hessian) {
  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream output;
  known_hessian_model_namespace::known_hessian_model known_hessian_model(
      data_var_context, 0, &output);

  Eigen::MatrixXd L(3, 3);
  Eigen::VectorXd q(3);
  L << 2.0, 0.0, 0.0, 0.7, 1.3, 0.0, -1.5, 2.0, 4.0;
  q << 0.0, 0.0, 0.0;

  double eval = stan::mcmc::internal::eigenvalue_scaled_hessian(
      known_hessian_model, L, q);

  EXPECT_LT(std::abs(eval - 22.8141075806892850) / eval, 1e-2);
}
