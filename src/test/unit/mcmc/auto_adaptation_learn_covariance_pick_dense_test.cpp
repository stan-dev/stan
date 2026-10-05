#include <stan/mcmc/auto_adaptation.hpp>
#include <test/test-models/good/model/correlated_gaussian.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <gtest/gtest.h>

TEST(McmcVarAdaptation, learn_covariance_pick_dense) {
  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream output;
  correlated_gaussian_model_namespace::correlated_gaussian_model
      correlated_gaussian_model(data_var_context, 0, &output);

  stan::test::unit::instrumented_logger logger;

  const int M = 2;
  const int N = 20;
  Eigen::MatrixXd qs(N, M);
  qs << 0.256173753306128, -0.0238087093098673, -1.63218152810157,
      -1.5309929638363, -0.812451331685826, -1.15062373620068,
      -1.49814775191801, -1.51310110681996, 0.738630631536685, 1.03588205799336,
      0.472288580035284, 0.250286770328584, -1.63634486169493, -1.6222798835089,
      -0.400790615207103, -0.337669147200631, -0.568779612417544,
      -0.424833495378187, 0.103690913176746, 0.272885200284842,
      -0.453017424229528, -0.504634004215693, 3.34484533887237,
      3.29418872328382, -1.3376507113241, -1.32724775403694, -0.137543235057544,
      -0.0290938109919368, -1.58194496352741, -1.39338740677379,
      0.312166136194586, 0.336989933768233, -0.628941448228566,
      -0.850758612234264, -0.766816808981044, -0.645020468024267,
      -0.75078110234827, -0.502544092120385, -0.00694807494461906,
      -0.186748159558166;

  Eigen::MatrixXd covar(M, M);
  bool covar_is_diagonal;

  Eigen::MatrixXd target_covar(M, M);

  target_covar << 1.0311414783609130, 1.0100577463968425, 1.0100577463968425,
      1.0148380697138280;

  stan::mcmc::auto_adaptation adapter(M);
  adapter.set_window_params(50, 0, 0, N, logger);

  for (int i = 0; i < N; ++i) {
    Eigen::VectorXd q = qs.block(i, 0, 1, M).transpose();
    adapter.learn_covariance(correlated_gaussian_model, covar,
                             covar_is_diagonal, q);
  }

  for (int i = 0; i < covar.size(); ++i) {
    EXPECT_FLOAT_EQ(target_covar(i), covar(i));
  }

  EXPECT_EQ(covar_is_diagonal, false);

  EXPECT_EQ(0, logger.call_count());
}
