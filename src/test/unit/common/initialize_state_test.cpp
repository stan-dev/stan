#include <ostream>
#include <stan/common/initialize_state.hpp>
#include <stan/model/prob_grad.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

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
                       Eigen::VectorXd& params_r__,
                       std::ostream* out) const {
    transform_inits_calls++;
    for (int n = 0; n < params_r__.size(); n++) {
      params_r__[n] = n;
    }
  }

  void unconstrained_param_names(std::vector<std::string>& param_names__,
                                 bool include_tparams__ = true,
                                 bool include_gqs__ = true) const {
    param_names__.clear();
    for (int n = 0; n < num_params_r__; n++) {
      std::stringstream param_name;
      param_name << "param_" << n;
      param_names__.push_back(param_name.str());
    }
  }
  
  mutable int templated_log_prob_calls;
  mutable int transform_inits_calls;
  double log_prob_return_value;
};

// Mock Inf Model returns inf, -inf
class mock_inf_model: public stan::model::prob_grad {
public:

  mock_inf_model(size_t num_params_r): 
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
                       Eigen::VectorXd& params_r__,
                       std::ostream* out) const {
    transform_inits_calls++;
    for (int n = 0; n < params_r__.size(); n++) {
      if (n % 2 == 0)
        params_r__[n] = std::numeric_limits<double>::infinity();
      else
        params_r__[n] = -std::numeric_limits<double>::infinity();
    }
  }

  void unconstrained_param_names(std::vector<std::string>& param_names__,
                                 bool include_tparams__ = true,
                                 bool include_gqs__ = true) const {
    param_names__.clear();
    for (int n = 0; n < num_params_r__; n++) {
      std::stringstream param_name;
      param_name << "param_" << n;
      param_names__.push_back(param_name.str());
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
    return calls / 10000.0;
  }

  static result_type max() {
    return 1.0;
  }

  static result_type min() {
    return -1.0;
  }

  int calls;
};

class mock_context_factory 
  : public stan::common::var_context_factory<stan::io::dump> {
public:
  mock_context_factory() 
    : calls(0),
      last_call("") { }
  
  void reset() {
    calls = 0;
    last_call = "";
  }
  
  stan::io::dump operator()(const std::string source) {
    calls++;
    last_call = source;
    std::string txt = "a <- 0\nb <- 1\nc <- 2";
    std::stringstream in(txt);
    return stan::io::dump(in);
  }
  
  int calls;
  std::string last_call;
};

class StanCommon : public testing::Test {
public:
  StanCommon() :
    model(3),
    inf_model(3) {}

  void SetUp() {
    cont_params = Eigen::VectorXd::Zero(3);
    model.reset();
    inf_model.reset();
    rng.reset();
    output.clear();
    context_factory.reset();
  }
  
  std::string init;
  Eigen::VectorXd cont_params;
  mock_model model;
  mock_inf_model inf_model;
  mock_rng rng;
  std::stringstream output;
  mock_context_factory context_factory;
};

TEST_F(StanCommon, initialize_state_0) {
  using stan::common::initialize_state;
  init = "0";
  EXPECT_TRUE(initialize_state(init,
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

TEST_F(StanCommon, initialize_state_zero) {
  using stan::common::initialize_state_zero;
  EXPECT_TRUE(initialize_state_zero(cont_params,
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
  
  EXPECT_FALSE(initialize_state_zero(cont_params,
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
  FAIL() << "need to add a test here";
}

TEST_F(StanCommon, initialize_state_number) {
  init = "1.5";
  using stan::common::initialize_state;
  EXPECT_TRUE(initialize_state(init,
                               cont_params,
                               model,
                               rng,
                               &output,
                               context_factory));
  ASSERT_EQ(3, cont_params.size());
  EXPECT_FLOAT_EQ((1.0 / 10000.0 / (rng.max() - rng.min())) * 3, cont_params[0]);
  EXPECT_FLOAT_EQ((2.0 / 10000.0 / (rng.max() - rng.min())) * 3, cont_params[1]);
  EXPECT_FLOAT_EQ((3.0 / 10000.0 / (rng.max() - rng.min())) * 3, cont_params[2]);
  EXPECT_EQ(1, model.templated_log_prob_calls);
  EXPECT_EQ(0, model.transform_inits_calls);
  EXPECT_EQ(3, rng.calls);
  EXPECT_EQ("", output.str());
  EXPECT_EQ(0, context_factory.calls);
  EXPECT_EQ("", context_factory.last_call);
}

TEST_F(StanCommon, initialize_state_random) {
  using stan::common::initialize_state_random;

  EXPECT_TRUE(initialize_state_random(1.5,
                                      cont_params,
                                      model,
                                      rng,
                                      &output));
  ASSERT_EQ(3, cont_params.size());
  EXPECT_FLOAT_EQ((1.0 / 10000.0 / (rng.max() - rng.min())) * 3, cont_params[0]);
  EXPECT_FLOAT_EQ((2.0 / 10000.0 / (rng.max() - rng.min())) * 3, cont_params[1]);
  EXPECT_FLOAT_EQ((3.0 / 10000.0 / (rng.max() - rng.min())) * 3, cont_params[2]);
  EXPECT_EQ(1, model.templated_log_prob_calls);
  EXPECT_EQ(0, model.transform_inits_calls);
  EXPECT_EQ(3, rng.calls);
  EXPECT_EQ("", output.str());
}


TEST_F(StanCommon, initialize_state_random_reject_all) {
  using stan::common::initialize_state_random;
  model.log_prob_return_value = -std::numeric_limits<double>::infinity();
  EXPECT_FALSE(initialize_state_random(1.5,
                                       cont_params,
                                       model,
                                       rng,
                                       &output));
  ASSERT_EQ(3, cont_params.size());
  EXPECT_FLOAT_EQ((298.0 / 10000.0 / (rng.max() - rng.min())) * 3, cont_params[0]);
  EXPECT_FLOAT_EQ((299.0 / 10000.0 / (rng.max() - rng.min())) * 3, cont_params[1]);
  EXPECT_FLOAT_EQ((300.0 / 10000.0 / (rng.max() - rng.min())) * 3, cont_params[2]);
  EXPECT_EQ(100, model.templated_log_prob_calls);
  EXPECT_EQ(0, model.transform_inits_calls);
  EXPECT_EQ(300, rng.calls);
  EXPECT_NE("", output.str()) << "expecting error message";
}


TEST_F(StanCommon, DISABLED_initialize_state_random_reject_handful) {
  // FIXME: write this test. Need to extend the mock model.
  FAIL() << "should check that it can recover after a handful of rejections";
}

TEST_F(StanCommon, initialize_state_string) {
  init = "abcd";
  using stan::common::initialize_state;
  EXPECT_TRUE(initialize_state(init,
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

TEST_F(StanCommon, initialize_state_source) {
  init = "abcd";
  using stan::common::initialize_state_source;
  EXPECT_TRUE(initialize_state_source(init,
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

TEST_F(StanCommon, initialize_state_source_neg_infinity) {
  init = "abcd";
  using stan::common::initialize_state_source;
  model.log_prob_return_value 
    = -std::numeric_limits<double>::infinity();
  EXPECT_FALSE(initialize_state_source(init,
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
  EXPECT_NE("", output.str())
    << "expecting some message here";
  EXPECT_EQ(1, context_factory.calls);
  EXPECT_EQ("abcd", context_factory.last_call);
}

TEST_F(StanCommon, DISABLED_initialize_state_source_gradient_throws) {
  // FIXME: add this test
  FAIL() << "it's not easy to force the model to throw when calculating the gradient"
         << " using this mock. need to add this test";
}

TEST_F(StanCommon, DISABLED_initialize_state_source_gradient_infinite) {
  // FIXME: add this test
  FAIL() << "it's not easy to force the model to set gradients "
         << " using this mock. need to add this test";
}

TEST_F(StanCommon, get_double_from_string) {
  using stan::common::get_double_from_string;
  double val;

  EXPECT_TRUE(get_double_from_string("0", val));
  EXPECT_FLOAT_EQ(0.0, val);

  EXPECT_TRUE(get_double_from_string("0.0", val));
  EXPECT_FLOAT_EQ(0.0, val);

  EXPECT_TRUE(get_double_from_string("123", val));
  EXPECT_FLOAT_EQ(123.0, val);

  EXPECT_FALSE(get_double_from_string("foo", val));
  EXPECT_PRED1(boost::math::isnan<double>,
               val);

  EXPECT_FALSE(get_double_from_string("0.0.0", val));
  EXPECT_PRED1(boost::math::isnan<double>,
               val);

  EXPECT_FALSE(get_double_from_string("", val));
  EXPECT_PRED1(boost::math::isnan<double>,
               val);
}

TEST_F(StanCommon, validate_unconstrained_initialization) {
  using stan::common::validate_unconstrained_initialization;
  Eigen::VectorXd valid(inf_model.num_params_r());
  valid.setZero();
  
  EXPECT_TRUE(validate_unconstrained_initialization(valid, inf_model));
  

  Eigen::VectorXd invalid(inf_model.num_params_r());
  for (int i = 0; i < invalid.size(); i++) {
    invalid.setZero();
    if (i % 2 == 0)
      invalid[i] = std::numeric_limits<double>::infinity();
    else
      invalid[i] = -std::numeric_limits<double>::infinity();
    
    std::stringstream expected_msg;

    expected_msg << "param_" << i << " initialized to invalid value ("
                 << invalid[i] << ")";

    EXPECT_THROW_MSG(validate_unconstrained_initialization(invalid, inf_model), 
                     std::invalid_argument,
                     expected_msg.str());
  }

  for (int i = 0; i < invalid.size(); i++) {
    invalid.setZero();
    invalid[i] = std::numeric_limits<double>::quiet_NaN();
    
    std::stringstream expected_msg;

    expected_msg << "param_" << i << " initialized to invalid value ("
                 << invalid[i] << ")";

    EXPECT_THROW_MSG(validate_unconstrained_initialization(invalid, inf_model), 
                     std::invalid_argument,
                     expected_msg.str());
  }
}

TEST_F(StanCommon, initialize_state_source_inf) {
  init = "abcd";
  using stan::common::initialize_state_source;
  EXPECT_FALSE(initialize_state_source(init,
                                       cont_params,
                                       inf_model,
                                       rng,
                                       &output,
                                       context_factory));
  ASSERT_EQ(3, cont_params.size());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), cont_params[0]);
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), cont_params[1]);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), cont_params[2]);
  EXPECT_EQ(0, inf_model.templated_log_prob_calls);
  EXPECT_EQ(1, inf_model.transform_inits_calls);
  EXPECT_EQ(0, rng.calls);
  std::stringstream expected_msg;
  expected_msg << "param_0 initialized to invalid value (" 
               << std::numeric_limits<double>::infinity() << ")\n";
  EXPECT_EQ(expected_msg.str(), output.str());
  EXPECT_EQ(1, context_factory.calls);
  EXPECT_EQ("abcd", context_factory.last_call);
}
