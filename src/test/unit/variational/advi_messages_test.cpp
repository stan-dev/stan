#include <test/test-models/good/variational/univariate_no_constraint.hpp>
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
    err_msg1 = "Informational Message: The maximum number of iterations is reached!";
    err_msg2 = "Informational Message: The ELBO at a previous iteration is larger than the ELBO upon convergence!";

    // Create mock data_var_context
    static const std::string DATA = "";
    std::stringstream data_stream(DATA);
    stan::io::dump data_var_context(data_stream);
    // data_stream.close();

    model_ = new stan_model(data_var_context, &model_stream_);
    cont_params_ = Eigen::VectorXd::Zero(model_->num_params_r());
    base_rng_.seed(3021828106u);
    model_stream_.str("");
    print_stream_.str("");
    output_stream_.str("");
    diagnostic_stream_.str("");

    advi_meanfield_ = new stan::variational::advi<stan_model, stan::variational::normal_meanfield, rng_t>
      (*model_, cont_params_,
       1, 100, 10.0,
       base_rng_,
       1, 1,
       &print_stream_,
       &output_stream_,
       &diagnostic_stream_);
    advi_fullrank_ = new stan::variational::advi<stan_model, stan::variational::normal_fullrank, rng_t>
      (*model_, cont_params_,
       1, 100, 10.0,
       base_rng_,
       1, 1,
       &print_stream_,
       &output_stream_,
       &diagnostic_stream_);
  }

  void TearDown() {
    delete advi_meanfield_;
    delete advi_fullrank_;
    delete model_;
  }

  std::string err_msg1;
  std::string err_msg2;

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


TEST_F(advi_test, max_iteration_warn_meanfield) {
  EXPECT_EQ(0, advi_meanfield_->run(0.01, 1));
  EXPECT_TRUE(print_stream_.str().find(err_msg1) != std::string::npos)
    << "The message should have err_msg1 inside it.";
}

TEST_F(advi_test, max_iteration_warn_fullrank) {
  EXPECT_EQ(0, advi_fullrank_->run(0.01, 1));
  EXPECT_TRUE(print_stream_.str().find(err_msg1) != std::string::npos)
    << "The message should have err_msg1 inside it.";
}

TEST_F(advi_test, prev_elbo_larger_meanfield) {
  EXPECT_EQ(0, advi_meanfield_->run(0.01, 10));
  EXPECT_TRUE(print_stream_.str().find(err_msg2) != std::string::npos)
    << "The message should have err_msg2 inside it.";
}

TEST_F(advi_test, prev_elbo_larger_fullrank) {
  EXPECT_EQ(0, advi_fullrank_->run(0.01, 10));
  EXPECT_TRUE(print_stream_.str().find(err_msg2) != std::string::npos)
    << "The message should have err_msg2 inside it.";
}
