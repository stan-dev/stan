#include <stan/common/initialize_state.hpp>
#include <stan/model/prob_grad.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

// Mock Model
class mock_model: public stan::model::prob_grad {
public:

  mock_model(size_t num_params_r): 
    stan::model::prob_grad(num_params_r),
    templated_log_prob_calls(0),
    transform_inits_calls(0),
    write_array_calls(0),
    log_prob_return_value(0.0) { }
  
  void reset() {
    templated_log_prob_calls = 0;
    transform_inits_calls = 0;
    write_array_calls = 0;
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
 
  void get_dims(std::vector<std::vector<size_t> >& dimss__) const {
    dimss__.resize(0);
    std::vector<size_t> scalar_dim;
    dimss__.push_back(scalar_dim);
    dimss__.push_back(scalar_dim);
    dimss__.push_back(scalar_dim);
  }

  void constrained_param_names(std::vector<std::string>& param_names__,
                               bool include_tparams__ = true,
                               bool include_gqs__ = true) const {
    param_names__.push_back("a");
    param_names__.push_back("b");
    param_names__.push_back("c");
  }

  template <typename RNG>
  void write_array(RNG& base_rng__,
                   std::vector<double>& params_r__,
                   std::vector<int>& params_i__,
                   std::vector<double>& vars__,
                   bool include_tparams__ = true,
                   bool include_gqs__ = true,
                   std::ostream* pstream__ = 0) const { 
    write_array_calls++;
     vars__.resize(0);
     for (size_t i = 0; i < params_r__.size(); i++)
       vars__.push_back(params_r__[i]);
  }
  
  mutable int templated_log_prob_calls;
  mutable int transform_inits_calls;
  mutable int write_array_calls;
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
  EXPECT_TRUE(initialize_state(init,
                               2,
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
  EXPECT_EQ(0, model.write_array_calls);
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
  EXPECT_EQ(0, model.write_array_calls);
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
  EXPECT_EQ(0, model.write_array_calls);
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
                               2,
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
  EXPECT_EQ(0, model.write_array_calls);
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
  EXPECT_EQ(0, model.write_array_calls);
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
  EXPECT_EQ(0, model.write_array_calls);
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
                               2,
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
  EXPECT_EQ(0, model.write_array_calls);
  EXPECT_EQ(0, rng.calls);
  EXPECT_EQ("", output.str());
  EXPECT_EQ(1, context_factory.calls);
  EXPECT_EQ("abcd", context_factory.last_call);
}

TEST_F(StanCommon, initialize_state_source) {
  init = "abcd";
  using stan::common::initialize_state_source;
  EXPECT_TRUE(initialize_state_source(init,
                                      2,
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
  EXPECT_EQ(0, model.write_array_calls);
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
                                      2,
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
  EXPECT_EQ(0, model.write_array_calls);
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

TEST(initialize_state, rm_indices_from_name) {
  std::string name1("alpha.1");
  std::vector<std::string> v;
  stan::common::rm_indices_from_name(v);
  EXPECT_EQ(0L, v.size());
  v.push_back(name1);
  stan::common::rm_indices_from_name(name1);
  EXPECT_STREQ("alpha", name1.c_str());
  v.push_back("alpha.2");
  v.push_back("beta.2");
  v.push_back("beta.3");
  stan::common::rm_indices_from_name(v);
  EXPECT_EQ(2L, v.size());

  std::string name2("gamma");
  stan::common::rm_indices_from_name(name2);
  EXPECT_STREQ("gamma", name1.c_str());


}

// Another complicated mock model with parameters as follows
/**
 * parameters {
 *   real alpha;
 *   real<lower=0> beta;
 *   vector[3] gamma;
 *   simplex[4] delta;
 * }
 */
class mock_model2: public mock_model {
public:

  mock_model2(): mock_model(8) {}
  void transform_inits(const stan::io::var_context& context__,
                       Eigen::VectorXd& params_r__) const {
    transform_inits_calls++;
    params_r__.resize(8);
    std::vector<double> rvec;
    if (!context__.contains_r("alpha") ||
        !context__.contains_r("beta") ||
        !context__.contains_r("gamma") ||
        !context__.contains_r("delta")) {
      throw std::runtime_error("missing some parameters");
    }
    rvec = context__.vals_r("alpha");
    params_r__[0] = rvec[0];
    rvec = context__.vals_r("beta");
    params_r__[1] = log(rvec[0]);
    rvec = context__.vals_r("gamma");
    params_r__[2] = rvec[0];
    params_r__[3] = rvec[1];
    params_r__[4] = rvec[2];
    rvec = context__.vals_r("delta");
    params_r__[5] = rvec[0];
    params_r__[6] = rvec[1];
    params_r__[7] = rvec[2];
  }
  void get_dims(std::vector<std::vector<size_t> >& dimss__) const {
    dimss__.resize(0);

    std::vector<size_t> scalar_dim;
    dimss__.push_back(scalar_dim);
    dimss__.push_back(scalar_dim);

    std::vector<size_t> vec_dim(1, 3);
    dimss__.push_back(vec_dim);
    vec_dim[0] = 4;
    dimss__.push_back(vec_dim);
  }
  void constrained_param_names(std::vector<std::string>& param_names__,
                               bool include_tparams__ = true,
                               bool include_gqs__ = true) const {
    param_names__.push_back("alpha");
    param_names__.push_back("beta");
    param_names__.push_back("gamma.1");
    param_names__.push_back("gamma.2");
    param_names__.push_back("gamma.3");
    param_names__.push_back("delta.1");
    param_names__.push_back("delta.2");
    param_names__.push_back("delta.3");
    param_names__.push_back("delta.4");
  }
  template <typename RNG>
  void write_array(RNG& base_rng__,
                   std::vector<double>& params_r__,
                   std::vector<int>& params_i__,
                   std::vector<double>& vars__,
                   bool include_tparams__ = true,
                   bool include_gqs__ = true,
                   std::ostream* pstream__ = 0) const {
    write_array_calls++;
    vars__.resize(0);
    vars__.push_back(params_r__[0]);
    vars__.push_back(exp(params_r__[1]));
    for (size_t i = 0; i < 6; i++)
      vars__.push_back(params_r__[i + 2]);
    vars__.push_back(1 - params_r__[5] - params_r__[6] - params_r__[7]);
  }
};

class mock_context_factory2: public mock_context_factory {
public:
  stan::io::dump operator()(const std::string source) {
    calls++;
    last_call = source;
    std::string txt = "alpha <- 0\n gamma <- c(1, 2, 3)";
    std::stringstream in(txt);
    return stan::io::dump(in);
  }
};

class StanCommon2 : public testing::Test {
public:
  StanCommon2() {}

  void SetUp() {
    cont_params = Eigen::VectorXd::Zero(8);
    model.reset();
    rng.reset();
    output.clear();
    context_factory.reset();
  }

  std::string init;
  Eigen::VectorXd cont_params;
  mock_model2 model;
  mock_rng rng;
  std::stringstream output;
  mock_context_factory2 context_factory;
};

TEST_F(StanCommon2, initialize_state_source_and_random) {
  init = "abcd";
  using stan::common::initialize_state_source_and_random;
  EXPECT_TRUE(initialize_state_source_and_random(init,
                                                 2,
                                                 cont_params,
                                                 model,
                                                 rng,
                                                 &output,
                                                 context_factory));
  ASSERT_EQ(8, cont_params.size());
  EXPECT_FLOAT_EQ(0, cont_params[0]);
  EXPECT_FLOAT_EQ((2.0 / 10000.0 / (rng.max() - rng.min())) * 4, cont_params[1]);
  EXPECT_FLOAT_EQ(1, cont_params[2]);
  EXPECT_FLOAT_EQ(2, cont_params[3]);
  EXPECT_FLOAT_EQ(3, cont_params[4]);
  EXPECT_FLOAT_EQ((6.0 / 10000.0 / (rng.max() - rng.min())) * 4, cont_params[5]);
  EXPECT_FLOAT_EQ((7.0 / 10000.0 / (rng.max() - rng.min())) * 4, cont_params[6]);
  EXPECT_FLOAT_EQ((8.0 / 10000.0 / (rng.max() - rng.min())) * 4, cont_params[7]);
  EXPECT_EQ(8, rng.calls);
  EXPECT_EQ(1, context_factory.calls);
  EXPECT_EQ(1, model.transform_inits_calls);
  EXPECT_EQ(1, model.write_array_calls);
  EXPECT_EQ("abcd", context_factory.last_call);
}


TEST_F(StanCommon2, are_all_pars_initialized) {
  EXPECT_FALSE(stan::common::are_all_pars_initialized(model, context_factory("a")));
  std::string txt = "alpha <- 0\nbeta <- 1\ngamma <- 2\ndelta <- 3";
  std::stringstream in(txt);
  stan::io::dump context(in);
  EXPECT_TRUE(stan::common::are_all_pars_initialized(model, context));
}

TEST_F(StanCommon2, initialize_state_source_and_random_R1) {
  init = "abcd";
  double R = 1;
  using stan::common::initialize_state_source_and_random;
  EXPECT_TRUE(initialize_state_source_and_random(init,
                                                 R,
                                                 cont_params,
                                                 model,
                                                 rng,
                                                 &output,
                                                 context_factory));
  ASSERT_EQ(8, cont_params.size());
  EXPECT_FLOAT_EQ(0, cont_params[0]);
  EXPECT_FLOAT_EQ((2.0 / 10000.0 / (rng.max() - rng.min())) * 2, cont_params[1]);
  EXPECT_FLOAT_EQ(1, cont_params[2]);
  EXPECT_FLOAT_EQ(2, cont_params[3]);
  EXPECT_FLOAT_EQ(3, cont_params[4]);
  EXPECT_FLOAT_EQ((6.0 / 10000.0 / (rng.max() - rng.min())) * 2, cont_params[5]);
  EXPECT_FLOAT_EQ((7.0 / 10000.0 / (rng.max() - rng.min())) * 2, cont_params[6]);
  EXPECT_FLOAT_EQ((8.0 / 10000.0 / (rng.max() - rng.min())) * 2, cont_params[7]);
  EXPECT_EQ(8, rng.calls);
  EXPECT_EQ(1, context_factory.calls);
  EXPECT_EQ(1, model.transform_inits_calls);
  EXPECT_EQ(1, model.write_array_calls);
  EXPECT_EQ("abcd", context_factory.last_call);
}

TEST_F(StanCommon2, initialize_state_disable_random_init) {
  using stan::common::initialize_state;
  init = "abcd";
  EXPECT_FALSE(initialize_state(init,
                                2,
                                cont_params,
                                model,
                                rng,
                                &output,
                                context_factory,
                                false));
  EXPECT_EQ(0, rng.calls);
  EXPECT_EQ(1, context_factory.calls);
  EXPECT_EQ("abcd", context_factory.last_call);
  EXPECT_EQ(1, model.transform_inits_calls);
  EXPECT_EQ(0, model.write_array_calls);
  EXPECT_NE("", output.str())
    << "expecting an error message here";
}
