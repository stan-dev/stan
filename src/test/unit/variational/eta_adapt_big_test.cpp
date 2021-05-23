#include <test/test-models/good/variational/eta_should_be_big.hpp>
#include <stan/variational/advi.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <boost/random/additive_combine.hpp>  // L'Ecuyer RNG

typedef boost::ecuyer1988 rng_t;

class eta_adapt_big_test : public ::testing::Test {
 public:
  eta_adapt_big_test()
      : logger(log_stream_, log_stream_, log_stream_, log_stream_,
               log_stream_) {}

  void SetUp() {
    static const std::string DATA = "";
    std::stringstream data_stream(DATA);
    stan::io::dump data_var_context(data_stream);

    model_ = new stan_model(data_var_context, 0, &model_stream_);
    cont_params_ = Eigen::VectorXd::Zero(model_->num_params_r());
    base_rng_.seed(927802408);
    model_stream_.str("");
    log_stream_.str("");

    advi_meanfield_ = new stan::variational::advi<
        stan_model, stan::variational::normal_meanfield, rng_t>(
        *model_, cont_params_, base_rng_, 1, 100, 100, 1);

    advi_fullrank_ = new stan::variational::advi<
        stan_model, stan::variational::normal_fullrank, rng_t>(
        *model_, cont_params_, base_rng_, 1, 100, 100, 1);

    advi_lowrank_ = new stan::variational::advi_lowrank<stan_model, rng_t>(
        *model_, cont_params_, base_rng_, 1, 1, 100, 100, 1);
  }

  void TearDown() {
    delete advi_meanfield_;
    delete advi_fullrank_;
    delete advi_lowrank_;
    delete model_;
  }

  stan::variational::advi<stan_model, stan::variational::normal_meanfield,
                          rng_t> *advi_meanfield_;
  stan::variational::advi<stan_model, stan::variational::normal_fullrank, rng_t>
      *advi_fullrank_;
  stan::variational::advi_lowrank<stan_model, rng_t> *advi_lowrank_;
  std::stringstream model_stream_;
  std::stringstream log_stream_;
  stan::callbacks::stream_logger logger;

  stan_model *model_;
  rng_t base_rng_;
  Eigen::VectorXd cont_params_;
};

TEST_F(eta_adapt_big_test, eta_should_be_big) {
  stan::variational::normal_meanfield meanfield_init
      = stan::variational::normal_meanfield(cont_params_);
  stan::variational::normal_fullrank fullrank_init
      = stan::variational::normal_fullrank(cont_params_);
  stan::variational::normal_lowrank lowrank_init
      = stan::variational::normal_lowrank(cont_params_, 1);

  EXPECT_EQ(100.0, advi_meanfield_->adapt_eta(meanfield_init, 50, logger));
  EXPECT_EQ(100.0, advi_fullrank_->adapt_eta(fullrank_init, 50, logger));
  EXPECT_EQ(100.0, advi_lowrank_->adapt_eta(lowrank_init, 50, logger));
}
