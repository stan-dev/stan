#include <test/test-models/good/variational/gradient_warn.hpp>
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
    std::fstream data_stream("src/test/test-models/good/variational/gradient_warn.data.R",
                             std::fstream::in);
    stan::io::dump data_var_context(data_stream);
    data_stream.close();

    model_ = new stan_model(data_var_context, &model_stream_);
    cont_params_ = Eigen::VectorXd::Zero(model_->num_params_r());
    base_rng_.seed(3021828106);
    model_stream_.str("");
    print_stream_.str("");
    output_stream_.str("");
    diagnostic_stream_.str("");

    advi_meanfield_ = new stan::variational::advi<stan_model, stan::variational::normal_meanfield, rng_t>
      (*model_, cont_params_,
       2, 100, 0.1,
       base_rng_,
       100, 1,
       &print_stream_,
       &output_stream_,
       &diagnostic_stream_);
    advi_fullrank_ = new stan::variational::advi<stan_model, stan::variational::normal_fullrank, rng_t>
      (*model_, cont_params_,
       1, 100, 0.1,
       base_rng_,
       100, 1,
       &print_stream_,
       &output_stream_,
       &diagnostic_stream_);
  }

  void TearDown() {
    delete advi_meanfield_;
    delete advi_fullrank_;
    delete model_;
  }

  stan::variational::advi<stan_model, stan::variational::normal_meanfield, rng_t> *advi_meanfield_;
  stan::variational::advi<stan_model, stan::variational::normal_fullrank, rng_t> *advi_fullrank_;
  std::stringstream model_stream_;
  std::stringstream print_stream_;
  std::stringstream output_stream_;
  std::stringstream diagnostic_stream_;

private:
  stan_model *model_;
  rng_t base_rng_;
  Eigen::VectorXd cont_params_;
};


TEST_F(advi_test, gradient_warn_meanfield) {
  EXPECT_EQ(0, advi_meanfield_->run(0.01, 10000));
  SUCCEED() << "expecting it to compile and run without problems";
}

TEST_F(advi_test, gradient_warn_fullrank) {
  EXPECT_EQ(0, advi_fullrank_->run(0.01, 10000));
  SUCCEED() << "expecting it to compile and run without problems";
}
