#include <stan/common/initialize_state.hpp>
#include <stan/model/prob_grad.hpp>
#include <gtest/gtest.h>

// Mock Model
class mock_model: public stan::model::prob_grad {
public:

  mock_model(size_t num_params_r): 
    stan::model::prob_grad(num_params_r),
    templated_log_prob_calls(0),
    transform_inits_calls(0),
    log_prob_return_value(0.0) { }
  
  void reset() {
    templated_log_prob_calls = 0;
    transform_inits_calls = 0;
    log_prob_return_value = 0.0;
  }

  template <bool propto, bool jacobian_adjust_transforms, typename T>
  T log_prob(Eigen::Matrix<T,Eigen::Dynamic,1>& params_r,
             std::ostream* output_stream = 0) const {
    templated_log_prob_calls++;
    return log_prob_return_value;
  }
  
  void transform_inits(const stan::io::var_context& context__,
                       Eigen::VectorXd& params_r__) const {
    transform_inits_calls++;
    for (int n = 0; n < params_r__.size(); n++) {
      params_r__[n] = n;
    }
  }
  
  mutable int templated_log_prob_calls;
  mutable int transform_inits_calls;
  double log_prob_return_value;
};

class mock_rng {
public:
  typedef double result_type;

  mock_rng() :
    calls(0) { }
    
  void reset() {
    calls = 0;
  }

  result_type operator()() {
    calls++;
    return calls;
  }

  static result_type max() {
    return 5;
  }

  static result_type min() {
    return -5;
  }

  int calls;
};

class mock_context_factory : public stan::common::var_context_factory {
public:
  mock_context_factory() 
    : calls(0),
      last_call("") { }
  
  void reset() {
    calls = 0;
    last_call = "";
  }
  
  stan::io::var_context* operator()(const std::string source) {
    calls++;
    last_call = source;
    std::string txt = "a <- 0\nb <- 1\nc <- 2";
    std::stringstream in(txt);
    return new stan::io::dump(in);
  }
  
  int calls;
  std::string last_call;
};

class StanCommon : public testing::Test {
public:
  StanCommon() :
    model(3) {}

  void SetUp() {
    cont_params = Eigen::VectorXd::Zero(3);
    model.reset();
    rng.reset();
    output.clear();
    context_factory.reset();
  }
  
  std::string init;
  Eigen::VectorXd cont_params;
  mock_model model;
  mock_rng rng;
  std::stringstream output;
  mock_context_factory context_factory;
};

TEST_F(StanCommon, initialize_state_0) {
  using stan::common::initialize_state;
  init = "0";
  EXPECT_NO_THROW(initialize_state(init,
                                   cont_params,
                                   model,
                                   rng,
                                   &output,
                                   context_factory));
  ASSERT_EQ(3, cont_params.size());
  EXPECT_FLOAT_EQ(0, cont_params[0]);
  EXPECT_FLOAT_EQ(0, cont_params[1]);
  EXPECT_FLOAT_EQ(0, cont_params[2]);
  EXPECT_EQ(1, model.templated_log_prob_calls);
  EXPECT_EQ(0, model.transform_inits_calls);
  EXPECT_EQ(0, rng.calls);
  EXPECT_EQ("", output.str());
  EXPECT_EQ(0, context_factory.calls);
  EXPECT_EQ("", context_factory.last_call);
}

TEST_F(StanCommon, initialize_state_zero_regular_execution) {
  using stan::common::initialize_state_zero;
  EXPECT_NO_THROW(initialize_state_zero(cont_params,
                                        model,
                                        &output));
  ASSERT_EQ(3, cont_params.size());
  EXPECT_FLOAT_EQ(0, cont_params[0]);
  EXPECT_FLOAT_EQ(0, cont_params[1]);
  EXPECT_FLOAT_EQ(0, cont_params[2]);
  EXPECT_EQ(1, model.templated_log_prob_calls);
  EXPECT_EQ(0, model.transform_inits_calls);
  EXPECT_EQ("", output.str());
}

TEST_F(StanCommon, initialize_state_zero_negative_infinity) {
  using stan::common::initialize_state_zero;
  model.log_prob_return_value = 
    -std::numeric_limits<double>::infinity();
  
  EXPECT_NO_THROW(initialize_state_zero(cont_params,
                                        model,
                                        &output));
  ASSERT_EQ(3, cont_params.size());
  EXPECT_FLOAT_EQ(0, cont_params[0]);
  EXPECT_FLOAT_EQ(0, cont_params[1]);
  EXPECT_FLOAT_EQ(0, cont_params[2]);
  EXPECT_EQ(1, model.templated_log_prob_calls);
  EXPECT_EQ(0, model.transform_inits_calls);
  EXPECT_NE("", output.str()) 
    << "expecting an error message here";
}

TEST_F(StanCommon, DISABLED_initialize_state_zero_grad_error) {
  // FIXME: it's really hard to get the derivatives to be off
  //        through mock objects
}

TEST_F(StanCommon, initialize_state_number) {
  init = "1.5";
  EXPECT_NO_THROW(initialize_state(init,
                                   cont_params,
                                   model,
                                   rng,
                                   &output,
                                   context_factory));
  ASSERT_EQ(3, cont_params.size());
  EXPECT_FLOAT_EQ((1.0 / 10) * 3, cont_params[0]);
  EXPECT_FLOAT_EQ((2.0 / 10) * 3, cont_params[1]);
  EXPECT_FLOAT_EQ((3.0 / 10) * 3, cont_params[2]);
  EXPECT_EQ(1, model.templated_log_prob_calls);
  EXPECT_EQ(0, model.transform_inits_calls);
  EXPECT_EQ(3, rng.calls);
  EXPECT_EQ("", output.str());
  EXPECT_EQ(0, context_factory.calls);
  EXPECT_EQ("", context_factory.last_call);
}

TEST_F(StanCommon, initialize_state_string) {
  init = "abcd";
  EXPECT_NO_THROW(initialize_state(init,
                                   cont_params,
                                   model,
                                   rng,
                                   &output,
                                   context_factory));
  ASSERT_EQ(3, cont_params.size());
  EXPECT_FLOAT_EQ(0, cont_params[0]);
  EXPECT_FLOAT_EQ(1, cont_params[1]);
  EXPECT_FLOAT_EQ(2, cont_params[2]);
  EXPECT_EQ(1, model.templated_log_prob_calls);
  EXPECT_EQ(1, model.transform_inits_calls);
  EXPECT_EQ(0, rng.calls);
  EXPECT_EQ("", output.str());
  EXPECT_EQ(1, context_factory.calls);
  EXPECT_EQ("abcd", context_factory.last_call);
}
