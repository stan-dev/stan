#include <test/test-models/good/variational/eta_should_be_small.hpp>
#include <stan/variational/advi.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/io/empty_var_context.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <stan/services/util/create_rng.hpp>
#include <boost/version.hpp>

class eta_adapt_small_test : public ::testing::Test {
 public:
  eta_adapt_small_test()
      : logger(log_stream_, log_stream_, log_stream_, log_stream_,
               log_stream_) {}

  void SetUp() {
    stan::io::empty_var_context data_var_context;

    model_ = new stan_model(data_var_context, 0, &model_stream_);
    cont_params_ = Eigen::VectorXd::Zero(model_->num_params_r());
    base_rng_.seed(727802409);
    model_stream_.str("");
    log_stream_.str("");

    advi_meanfield_ = new stan::variational::advi<
        stan_model, stan::variational::normal_meanfield, stan::rng_t>(
        *model_, cont_params_, base_rng_, 1, 100, 100, 1);

    advi_fullrank_ = new stan::variational::advi<
        stan_model, stan::variational::normal_fullrank, stan::rng_t>(
        *model_, cont_params_, base_rng_, 1, 100, 100, 1);
  }

  void TearDown() {
    delete advi_meanfield_;
    delete advi_fullrank_;
    delete model_;
  }

  stan::variational::advi<stan_model, stan::variational::normal_meanfield,
                          stan::rng_t> *advi_meanfield_;
  stan::variational::advi<stan_model, stan::variational::normal_fullrank,
                          stan::rng_t> *advi_fullrank_;
  std::stringstream model_stream_;
  std::stringstream log_stream_;
  stan::callbacks::stream_logger logger;

  stan_model *model_;
  stan::rng_t base_rng_;
  Eigen::VectorXd cont_params_;
};

TEST_F(eta_adapt_small_test, eta_should_be_small) {
  stan::variational::normal_meanfield meanfield_init
      = stan::variational::normal_meanfield(cont_params_);
  stan::variational::normal_fullrank fullrank_init
      = stan::variational::normal_fullrank(cont_params_);

  EXPECT_EQ(0.1, advi_meanfield_->adapt_eta(meanfield_init, 1000, logger));
  EXPECT_EQ(0.1, advi_fullrank_->adapt_eta(fullrank_init, 1000, logger));
}
