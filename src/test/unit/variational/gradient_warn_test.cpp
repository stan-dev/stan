#include <test/test-models/good/variational/gradient_warn.hpp>
#include <stan/variational/advi.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG

typedef boost::ecuyer1988 rng_t;

class advi_test : public ::testing::Test {
public:
  advi_test()
    : logger(message_stream_, message_stream_, message_stream_, message_stream_, message_stream_),
      parameter_writer(parameter_stream_),
      diagnostic_writer(diagnostic_stream_) { }

  void SetUp() {
    std::fstream data_stream("src/test/test-models/good/variational/gradient_warn.data.R",
                             std::fstream::in);
    stan::io::dump data_var_context(data_stream);
    data_stream.close();

    model_ = new stan_model(data_var_context, &model_stream_);
    cont_params_ = Eigen::VectorXd::Zero(model_->num_params_r());
    base_rng_.seed(3021828106u);
    model_stream_.str("");
    message_stream_.str("");
    parameter_stream_.str("");
    diagnostic_stream_.str("");

    advi_meanfield_ = new stan::variational::advi<stan_model, stan::variational::normal_meanfield, rng_t>
      (*model_, cont_params_, base_rng_,
       1, 100,
       100, 1);
    advi_fullrank_ = new stan::variational::advi<stan_model, stan::variational::normal_fullrank, rng_t>
      (*model_, cont_params_, base_rng_,
       1, 100,
       100, 1);
      }

  void TearDown() {
    delete advi_meanfield_;
    delete advi_fullrank_;
    delete model_;
  }

  stan::variational::advi<stan_model, stan::variational::normal_meanfield, rng_t> *advi_meanfield_;
  stan::variational::advi<stan_model, stan::variational::normal_fullrank, rng_t> *advi_fullrank_;
  std::stringstream model_stream_;
  std::stringstream message_stream_;
  std::stringstream parameter_stream_;
  std::stringstream diagnostic_stream_;

  stan::callbacks::stream_logger logger;
  stan::callbacks::stream_writer parameter_writer;
  stan::callbacks::stream_writer diagnostic_writer;

private:
  stan_model *model_;
  rng_t base_rng_;
  Eigen::VectorXd cont_params_;
};


TEST_F(advi_test, gradient_warn_meanfield) {
  EXPECT_EQ(0, advi_meanfield_->run(0.1, false, 50, 0.01, 10000,
                                    logger,
                                    parameter_writer,
                                    diagnostic_writer));
  SUCCEED() << "expecting it to compile and run without problems";
}

TEST_F(advi_test, gradient_warn_fullrank) {
  EXPECT_EQ(0, advi_fullrank_->run(0.1, false, 50, 0.01, 10000,
                                   logger,
                                   parameter_writer,
                                   diagnostic_writer));
  SUCCEED() << "expecting it to compile and run without problems";
}
