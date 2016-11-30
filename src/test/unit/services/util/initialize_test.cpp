#include <stan/services/util/initialize.hpp>
#include <gtest/gtest.h>
#include <stan/callbacks/stream_writer.hpp>
#include <sstream>
#include <test/test-models/good/services/test_lp.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/services/util/create_rng.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>

class ServicesUtilInitialize : public testing::Test {
public:
  ServicesUtilInitialize()
    : model(empty_context, &model_ss),
      message(message_ss),
      rng(stan::services::util::create_rng(0, 1)) {}

  stan::io::empty_var_context empty_context;
  std::stringstream model_ss, message_ss;
  stan_model model;
  stan::callbacks::stream_writer message;
  stan::test::unit::instrumented_writer init;
  boost::ecuyer1988 rng;
};

TEST_F(ServicesUtilInitialize, radius_zero_print_false) {
  std::vector<double> params;
    
  double init_radius = 0;
  bool print_timing = false;
  params = stan::services::util::initialize(model, empty_context, rng,
                                            init_radius, print_timing,
                                            message, init);
  ASSERT_EQ(model.num_params_r(), params.size())
    << "2 parameters";
  EXPECT_FLOAT_EQ(0, params[0]);
  EXPECT_FLOAT_EQ(0, params[1]);

  EXPECT_EQ("", message_ss.str());
  ASSERT_EQ(1, init.vector_double_values().size());
  ASSERT_EQ(2, init.vector_double_values()[0].size());
  EXPECT_EQ(params[0], init.vector_double_values()[0][0]);
  EXPECT_EQ(params[1], init.vector_double_values()[0][1]);
}
  
TEST_F(ServicesUtilInitialize, radius_zero_print_true) {
  std::vector<double> params;
    
  double init_radius = 0;
  bool print_timing = true;
  params = stan::services::util::initialize(model, empty_context, rng,
                                            init_radius, print_timing,
                                            message, init);
  ASSERT_EQ(model.num_params_r(), params.size())
    << "2 parameters";
  EXPECT_FLOAT_EQ(0, params[0]);
  EXPECT_FLOAT_EQ(0, params[1]);

  EXPECT_TRUE(message_ss.str().find("Gradient evaluation") != std::string::npos);
  ASSERT_EQ(1, init.vector_double_values().size());
  ASSERT_EQ(2, init.vector_double_values()[0].size());
  EXPECT_EQ(params[0], init.vector_double_values()[0][0]);
  EXPECT_EQ(params[1], init.vector_double_values()[0][1]);
}

TEST_F(ServicesUtilInitialize, radius_two_print_false) {
  std::vector<double> params;
    
  double init_radius = 2;
  bool print_timing = false;
  params = stan::services::util::initialize(model, empty_context, rng,
                                            init_radius, print_timing,
                                            message, init);
  ASSERT_EQ(model.num_params_r(), params.size())
    << "2 parameters";
  EXPECT_GT(params[0], -init_radius);
  EXPECT_LT(params[0], init_radius);
  EXPECT_GT(params[1], -init_radius);
  EXPECT_LT(params[1], init_radius);

  EXPECT_EQ("", message_ss.str());
  ASSERT_EQ(1, init.vector_double_values().size());
  ASSERT_EQ(2, init.vector_double_values()[0].size());
  EXPECT_EQ(params[0], init.vector_double_values()[0][0]);
  EXPECT_EQ(params[1], init.vector_double_values()[0][1]);
}

TEST_F(ServicesUtilInitialize, radius_two_print_true) {
  std::vector<double> params;
    
  double init_radius = 2;
  bool print_timing = true;
  params = stan::services::util::initialize(model, empty_context, rng,
                                            init_radius, print_timing,
                                            message, init);
  ASSERT_EQ(model.num_params_r(), params.size())
    << "2 parameters";
  EXPECT_GT(params[0], -init_radius);
  EXPECT_LT(params[0], init_radius);
  EXPECT_GT(params[1], -init_radius);
  EXPECT_LT(params[1], init_radius);

  EXPECT_TRUE(message_ss.str().find("Gradient evaluation") != std::string::npos);
  ASSERT_EQ(1, init.vector_double_values().size());
  ASSERT_EQ(2, init.vector_double_values()[0].size());
  EXPECT_EQ(params[0], init.vector_double_values()[0][0]);
  EXPECT_EQ(params[1], init.vector_double_values()[0][1]);
}

TEST_F(ServicesUtilInitialize, full_init_print_false) {
  std::vector<std::string> names_r;
  std::vector<double> values_r;
  std::vector<std::vector<size_t> > dim_r;
  names_r.push_back("y");
  values_r.push_back(6.35149);   // 1.5 unconstrained: -10 + 20 * inv.logit(1.5)
  values_r.push_back(-2.449187); // -0.5 unconstrained 
  std::vector<size_t> d;
  d.push_back(2);
  dim_r.push_back(d);
  stan::io::array_var_context init_context(names_r, values_r, dim_r);


  std::vector<double> params;
    
  double init_radius = 2;
  bool print_timing = false;
  params = stan::services::util::initialize(model, init_context, rng,
                                            init_radius, print_timing,
                                            message, init);
  ASSERT_EQ(model.num_params_r(), params.size())
    << "2 parameters";
  EXPECT_FLOAT_EQ(1.5, params[0]);
  EXPECT_FLOAT_EQ(-0.5, params[1]);

  EXPECT_EQ("", message_ss.str());
  ASSERT_EQ(1, init.vector_double_values().size());
  ASSERT_EQ(2, init.vector_double_values()[0].size());
  EXPECT_EQ(params[0], init.vector_double_values()[0][0]);
  EXPECT_EQ(params[1], init.vector_double_values()[0][1]);
}

TEST_F(ServicesUtilInitialize, full_init_print_true) {
  std::vector<std::string> names_r;
  std::vector<double> values_r;
  std::vector<std::vector<size_t> > dim_r;
  names_r.push_back("y");
  values_r.push_back(6.35149);   // 1.5 unconstrained: -10 + 20 * inv.logit(1.5)
  values_r.push_back(-2.449187); // -0.5 unconstrained 
  std::vector<size_t> d;
  d.push_back(2);
  dim_r.push_back(d);
  stan::io::array_var_context init_context(names_r, values_r, dim_r);


  std::vector<double> params;
    
  double init_radius = 2;
  bool print_timing = true;
  params = stan::services::util::initialize(model, init_context, rng,
                                            init_radius, print_timing,
                                            message, init);
  ASSERT_EQ(model.num_params_r(), params.size())
    << "2 parameters";
  EXPECT_FLOAT_EQ(1.5, params[0]);
  EXPECT_FLOAT_EQ(-0.5, params[1]);

  EXPECT_TRUE(message_ss.str().find("Gradient evaluation") != std::string::npos);
  ASSERT_EQ(1, init.vector_double_values().size());
  ASSERT_EQ(2, init.vector_double_values()[0].size());
  EXPECT_EQ(params[0], init.vector_double_values()[0][0]);
  EXPECT_EQ(params[1], init.vector_double_values()[0][1]);
}

namespace test {
  // Mock Throwing Model throws exception
  class mock_throwing_model: public stan::model::prob_grad {
  public:

    mock_throwing_model():
      stan::model::prob_grad(1),
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

    template <bool propto_, bool jacobian_, typename T_>
    T_ log_prob(std::vector<T_>& params_r_,
                 std::vector<int>& params_i_,
                 std::ostream* pstream_ = 0) const {
      ++templated_log_prob_calls;
      throw std::domain_error("throwing within log_prob");
      return log_prob_return_value;
    }

    void transform_inits(const stan::io::var_context& context_,
                         std::vector<int>& params_i_,
                         std::vector<double>& params_r_,
                         std::ostream* pstream_) const {
      ++transform_inits_calls;
      for (size_t n = 0; n < params_r_.size(); ++n) {
        params_r_[n] = n;
      }
    }

    void get_dims(std::vector<std::vector<size_t> >& dimss_) const {
      dimss_.resize(0);
      std::vector<size_t> scalar_dim;
      dimss_.push_back(scalar_dim);
    }

    void constrained_param_names(std::vector<std::string>& param_names_,
                                 bool include_tparams_ = true,
                                 bool include_gqs_ = true) const {
      param_names_.push_back("theta");
    }

    void get_param_names(std::vector<std::string>& names) const {
      constrained_param_names(names);
    }

    void unconstrained_param_names(std::vector<std::string>& param_names_,
                                   bool include_tparams_ = true,
                                   bool include_gqs_ = true) const {
      param_names_.clear();
      for (size_t n = 0; n < num_params_r_; ++n) {
        std::stringstream param_name;
        param_name << "param_" << n;
        param_names_.push_back(param_name.str());
      }
    }
    template <typename RNG>
    void write_array(RNG& base_rng_,
                     std::vector<double>& params_r_,
                     std::vector<int>& params_i_,
                     std::vector<double>& vars_,
                     bool include_tparams_ = true,
                     bool include_gqs_ = true,
                     std::ostream* pstream_ = 0) const {
      ++write_array_calls;
      vars_.resize(0);
      for (size_t i = 0; i < params_r_.size(); ++i)
        vars_.push_back(params_r_[i]);
    }

    mutable int templated_log_prob_calls;
    mutable int transform_inits_calls;
    mutable int write_array_calls;
    double log_prob_return_value;
  };
}

TEST_F(ServicesUtilInitialize, model_throws) {
  std::vector<double> params;
  test::mock_throwing_model throwing_model;
    
  double init_radius = 2;
  bool print_timing = false;
  EXPECT_THROW(params
               = stan::services::util::initialize(throwing_model, empty_context, rng,
                                                  init_radius, print_timing,
                                                  message, init),
               std::domain_error);
}
