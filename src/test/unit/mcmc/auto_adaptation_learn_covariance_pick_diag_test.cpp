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
  qs << 0.607446257145326, 0.338465765807058, 1.47389672467345,
      -1.0577986841911, 1.02886652895522, 0.364277500948572, 0.492316893603469,
      2.19693408641558, -0.931854393410476, 1.62634580968769,
      -0.443145375724188, 0.902790875582656, 0.517782110245233,
      -1.56724331755861, -1.7556390097031, 0.310274990315213,
      0.0394975482340945, 0.366999438969482, 1.29372950054929,
      0.361369734821582, -0.258301497542829, 0.166994731172984,
      0.492639248874412, -0.659502589885556, 0.913729457222598,
      1.99580706461809, 0.669655370469707, -0.509028392475839,
      -0.626041244059129, -0.771981104624195, -0.842385483586737,
      0.337166271031201, 0.548177804329155, -0.0462961925005498,
      0.955748803092952, 1.3141117316189, 0.335670079140694, 1.09112083087171,
      0.759245358940033, -1.11318882201676;

  Eigen::MatrixXd covar(M, M);
  bool covar_is_diagonal;

  Eigen::MatrixXd target_covar(M, M);

  target_covar << 0.55350038163333048, 0.0, 0.0, 0.86122545968912112;

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
