#include <test/test-models/good/variational/hier_logistic.hpp>
#include <stan/variational/advi.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG

typedef boost::ecuyer1988 rng_t;

class advi_test : public ::testing::Test {
public:
  void SetUp() {
    // Create mock data_var_context
    std::fstream data_stream("src/test/test-models/good/variational/hier_logistic.data.R",
                             std::fstream::in);
    stan::io::dump data_var_context(data_stream);
    data_stream.close();

    model_ = new stan_model(data_var_context, &model_stream_);
    model_null_stream_ = new stan_model(data_var_context, NULL);

    base_rng_.seed(0);
    cont_params_ = Eigen::VectorXd::Zero(model_->num_params_r());

    model_stream_.str("");
    print_stream_.str("");
    output_stream_.str("");
    diagnostic_stream_.str("");

    advi_ = new stan::variational::advi<stan_model, rng_t, stan::variational::normal_meanfield, stan::variational::normal_fullrank>
      (*model_, cont_params_,
       10, 100, 0.01,
       base_rng_,
       100, 1,
       &print_stream_,
       &output_stream_,
       &diagnostic_stream_);
    advi_null_streams_ = new stan::variational::advi<stan_model, rng_t, stan::variational::normal_meanfield, stan::variational::normal_fullrank>
      (*model_null_stream_, cont_params_,
       10, 100, 0.01,
       base_rng_,
       100, 1,
       NULL,
       NULL,
       NULL);
  }

  void TearDown() {
    delete advi_;
    delete model_;
    delete advi_null_streams_;
    delete model_null_stream_;
  }

  stan::variational::advi<stan_model, rng_t, stan::variational::normal_meanfield, stan::variational::normal_fullrank> *advi_;
  stan::variational::advi<stan_model, rng_t, stan::variational::normal_meanfield, stan::variational::normal_fullrank> *advi_null_streams_;
  std::stringstream model_stream_;
  std::stringstream print_stream_;
  std::stringstream output_stream_;
  std::stringstream diagnostic_stream_;

private:
  stan_model *model_;
  stan_model *model_null_stream_;
  rng_t base_rng_;
  Eigen::VectorXd cont_params_;
};


TEST_F(advi_test, hier_logistic_constraint_meanfield) {
  EXPECT_EQ(0, advi_->run_meanfield(1, 2e4));
  SUCCEED() << "expecting it to compile and run without problems";
}

TEST_F(advi_test, hier_logistic_constraint_fullrank_no_streams) {
  EXPECT_EQ(0, advi_null_streams_->run_meanfield(1, 2e4));
  SUCCEED() << "expecting it to compile and run without problems";
}

TEST_F(advi_test, hier_logistic_constraint_meanfield_no_streams) {
  EXPECT_EQ(0, advi_null_streams_->run_meanfield(1, 2e4));
  SUCCEED() << "expecting it to compile and run without problems";
}
