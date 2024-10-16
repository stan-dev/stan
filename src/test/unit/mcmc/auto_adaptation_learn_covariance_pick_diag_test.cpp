#include <stan/mcmc/auto_adaptation.hpp>
#include <test/test-models/good/model/independent_gaussian.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <gtest/gtest.h>

TEST(McmcVarAdaptation, learn_covariance_pick_diagonal) {
  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream output;
  independent_gaussian_model_namespace::independent_gaussian_model
      independent_gaussian_model(data_var_context, 0, &output);

  stan::test::unit::instrumented_logger logger;

  const int M = 2;
  const int N = 20;
  Eigen::MatrixXd qs(N, M);
  qs << -1.12310858, -0.98044486, -0.40288484, -0.31777537, -0.46665535,
      -0.41468940, 0.77996512, 0.79419535, -0.08336907, 0.12997631, 0.25331851,
      0.17888355, -0.02854676, -0.25660897, -0.04287046, 0.06199044, 1.36860228,
      1.16082198, -0.22577099, -0.27199475, 1.51647060, 1.46738068, -1.54875280,
      -1.42235482, 0.58461375, 0.40408060, 0.12385424, 0.12959917, 0.21594157,
      0.18045828, 0.37963948, 0.34225195, -0.50232345, -0.41356307, -0.33320738,
      -0.33695265, -1.01857538, -0.85228019, -1.07179123, -0.98666076;

  Eigen::MatrixXd covar(M, M);
  bool covar_is_diagonal;

  Eigen::MatrixXd target_covar(M, M);

  target_covar << 0.50172809066980716963, 0.0, 0.0, 0.41270751419364998247;

  stan::mcmc::auto_adaptation adapter(M);
  adapter.set_window_params(50, 0, 0, N, logger);

  for (int i = 0; i < N; ++i) {
    Eigen::VectorXd q = qs.block(i, 0, 1, M).transpose();
    adapter.learn_covariance(independent_gaussian_model, covar,
                             covar_is_diagonal, q);
  }

  for (int i = 0; i < covar.size(); ++i) {
    EXPECT_FLOAT_EQ(target_covar(i), covar(i));
  }

  EXPECT_EQ(covar_is_diagonal, true);

  EXPECT_EQ(0, logger.call_count());
}
