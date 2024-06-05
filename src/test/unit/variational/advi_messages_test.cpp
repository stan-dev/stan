#include <test/test-models/good/variational/univariate_no_constraint.hpp>
#include <stan/variational/advi.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/io/empty_var_context.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <stan/services/util/create_rng.hpp>

class advi_test : public ::testing::Test {
 public:
  advi_test()
      : logger(log_stream_, log_stream_, log_stream_, log_stream_, log_stream_),
        parameter_writer(parameter_stream_),
        diagnostic_writer(diagnostic_stream_) {}

  void SetUp() {
    err_msg1
        = "Informational Message: The maximum number of iterations is reached!";
    err_msg2
        = "Informational Message: The ELBO at a previous iteration is larger "
          "than the ELBO upon convergence!";

    stan::io::empty_var_context data_var_context;

    model_ = new stan_model(data_var_context, 0, &model_stream_);
    cont_params_ = Eigen::VectorXd::Zero(model_->num_params_r());
    base_rng_.seed(3021828109u);
    model_stream_.str("");
    log_stream_.str("");
    parameter_stream_.str("");
    diagnostic_stream_.str("");

    advi_meanfield_ = new stan::variational::advi<
        stan_model, stan::variational::normal_meanfield, stan::rng_t>(
        *model_, cont_params_, base_rng_, 1, 100, 1, 1);
    advi_fullrank_ = new stan::variational::advi<
        stan_model, stan::variational::normal_fullrank, stan::rng_t>(
        *model_, cont_params_, base_rng_, 1, 100, 1, 1);
  }

  void TearDown() {
    delete advi_meanfield_;
    delete advi_fullrank_;
    delete model_;
  }

  std::string err_msg1;
  std::string err_msg2;

  stan::variational::advi<stan_model, stan::variational::normal_meanfield,
                          stan::rng_t> *advi_meanfield_;
  stan::variational::advi<stan_model, stan::variational::normal_fullrank,
                          stan::rng_t> *advi_fullrank_;
  std::stringstream model_stream_;
  std::stringstream log_stream_;
  std::stringstream parameter_stream_;
  std::stringstream diagnostic_stream_;
  stan::callbacks::stream_logger logger;
  stan::callbacks::stream_writer parameter_writer;
  stan::callbacks::stream_writer diagnostic_writer;

 private:
  stan_model *model_;
  stan::rng_t base_rng_;
  Eigen::VectorXd cont_params_;
};

TEST_F(advi_test, max_iteration_warn_meanfield) {
  EXPECT_EQ(0, advi_meanfield_->run(10, 0, 50, 0.01, 1, logger,
                                    parameter_writer, diagnostic_writer));
  EXPECT_TRUE(log_stream_.str().find(err_msg1) != std::string::npos)
      << "The message should have err_msg1 inside it.";
}

TEST_F(advi_test, max_iteration_warn_fullrank) {
  EXPECT_EQ(0, advi_fullrank_->run(10, 0, 50, 0.01, 1, logger, parameter_writer,
                                   diagnostic_writer));
  EXPECT_TRUE(log_stream_.str().find(err_msg1) != std::string::npos)
      << "The message should have err_msg1 inside it.";
}

TEST_F(advi_test, prev_elbo_larger_meanfield) {
  EXPECT_EQ(0, advi_meanfield_->run(10, 0, 50, 0.1, 100, logger,
                                    parameter_writer, diagnostic_writer));
  EXPECT_TRUE(log_stream_.str().find(err_msg2) != std::string::npos)
      << "The message should have err_msg2 inside it.";
}

TEST_F(advi_test, prev_elbo_larger_fullrank) {
  EXPECT_EQ(0, advi_fullrank_->run(10, 0, 50, 0.2, 100, logger,
                                   parameter_writer, diagnostic_writer));
  EXPECT_TRUE(log_stream_.str().find(err_msg2) != std::string::npos)
      << "The message should have err_msg2 inside it.";
}
