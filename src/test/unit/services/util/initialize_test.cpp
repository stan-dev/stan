#include <stan/services/util/initialize.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <sstream>
#include <test/test-models/good/services/test_lp.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/services/util/create_rng.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>

class ServicesUtilInitialize : public testing::Test {
 public:
  ServicesUtilInitialize()
      : model(empty_context, 12345, &model_ss),
        message(message_ss),
        rng(stan::services::util::create_rng(0, 1)) {}

  stan_model model;
  stan::io::empty_var_context empty_context;
  std::stringstream model_ss;
  std::stringstream message_ss;
  stan::callbacks::stream_writer message;
  stan::test::unit::instrumented_logger logger;
  stan::test::unit::instrumented_writer init;
  boost::ecuyer1988 rng;
};

TEST_F(ServicesUtilInitialize, radius_zero__print_false) {
  std::vector<double> params;

  double init_radius = 0;
  bool print_timing = false;
  params = stan::services::util::initialize(model, empty_context, rng,
                                            init_radius, print_timing,
                                            logger, init);
  ASSERT_EQ(model.num_params_r(), params.size())
      << "2 parameters";
  EXPECT_FLOAT_EQ(0, params[0]);
  EXPECT_FLOAT_EQ(0, params[1]);

  EXPECT_EQ(0, logger.call_count());
  ASSERT_EQ(1, init.vector_double_values().size());
  ASSERT_EQ(2, init.vector_double_values()[0].size());
  EXPECT_EQ(params[0], init.vector_double_values()[0][0]);
  EXPECT_EQ(params[1], init.vector_double_values()[0][1]);
}

TEST_F(ServicesUtilInitialize, radius_zero__initialize_with_Jacobian) {
  std::vector<double> params;

  double init_radius = 0;
  bool print_timing = false;
  params = stan::services::util::initialize<false>(model, empty_context, rng,
                                                   init_radius, print_timing,
                                                   logger, init);
  ASSERT_EQ(model.num_params_r(), params.size())
      << "2 parameters";
  EXPECT_FLOAT_EQ(0, params[0]);
  EXPECT_FLOAT_EQ(0, params[1]);

  EXPECT_EQ(0, logger.call_count());
  ASSERT_EQ(1, init.vector_double_values().size());
  ASSERT_EQ(2, init.vector_double_values()[0].size());
  EXPECT_EQ(params[0], init.vector_double_values()[0][0]);
  EXPECT_EQ(params[1], init.vector_double_values()[0][1]);
}

TEST_F(ServicesUtilInitialize, radius_two__print_false) {
  std::vector<double> params;

  double init_radius = 2;
  bool print_timing = false;
  params = stan::services::util::initialize(model, empty_context, rng,
                                            init_radius, print_timing,
                                            logger, init);
  ASSERT_EQ(model.num_params_r(), params.size())
      << "2 parameters";
  EXPECT_GT(params[0], -init_radius);
  EXPECT_LT(params[0], init_radius);
  EXPECT_GT(params[1], -init_radius);
  EXPECT_LT(params[1], init_radius);

  EXPECT_EQ(0, logger.call_count());
  ASSERT_EQ(1, init.vector_double_values().size());
  ASSERT_EQ(2, init.vector_double_values()[0].size());
  EXPECT_EQ(params[0], init.vector_double_values()[0][0]);
  EXPECT_EQ(params[1], init.vector_double_values()[0][1]);
}

TEST_F(ServicesUtilInitialize, radius_two__print_true) {
  std::vector<double> params;

  double init_radius = 2;
  bool print_timing = true;
  params = stan::services::util::initialize(model, empty_context, rng,
                                            init_radius, print_timing,
                                            logger, init);
  ASSERT_EQ(model.num_params_r(), params.size())
      << "2 parameters";
  EXPECT_GT(params[0], -init_radius);
  EXPECT_LT(params[0], init_radius);
  EXPECT_GT(params[1], -init_radius);
  EXPECT_LT(params[1], init_radius);

  EXPECT_EQ(6, logger.call_count());
  EXPECT_EQ(6, logger.call_count_info());
  EXPECT_EQ(1, logger.find_info("Gradient evaluation"));
  ASSERT_EQ(1, init.vector_double_values().size());
  ASSERT_EQ(2, init.vector_double_values()[0].size());
  EXPECT_EQ(params[0], init.vector_double_values()[0][0]);
  EXPECT_EQ(params[1], init.vector_double_values()[0][1]);
}

TEST_F(ServicesUtilInitialize, full_init__print_false) {
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
                                            logger, init);
  ASSERT_EQ(model.num_params_r(), params.size())
      << "2 parameters";
  EXPECT_FLOAT_EQ(1.5, params[0]);
  EXPECT_FLOAT_EQ(-0.5, params[1]);

  EXPECT_EQ(0, logger.call_count());
  ASSERT_EQ(1, init.vector_double_values().size());
  ASSERT_EQ(2, init.vector_double_values()[0].size());
  EXPECT_EQ(params[0], init.vector_double_values()[0][0]);
  EXPECT_EQ(params[1], init.vector_double_values()[0][1]);
}

TEST_F(ServicesUtilInitialize, full_init__print_true) {
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
                                            logger, init);
  ASSERT_EQ(model.num_params_r(), params.size())
      << "2 parameters";
  EXPECT_FLOAT_EQ(1.5, params[0]);
  EXPECT_FLOAT_EQ(-0.5, params[1]);

  EXPECT_EQ(6, logger.call_count());
  EXPECT_EQ(6, logger.call_count_info());
  EXPECT_EQ(1, logger.find_info("Gradient evaluation"));
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

  template <bool propto__, bool jacobian__, typename T__>
  T__ log_prob(std::vector<T__>& params_r__,
               std::vector<int>& params_i__,
               std::ostream* pstream__ = 0) const {
    ++templated_log_prob_calls;
    throw std::domain_error("throwing within log_prob");
    return log_prob_return_value;
  }

  void transform_inits(const stan::io::var_context& context__,
                       std::vector<int>& params_i__,
                       std::vector<double>& params_r__,
                       std::ostream* pstream__) const {
    ++transform_inits_calls;
    for (size_t n = 0; n < params_r__.size(); ++n) {
      params_r__[n] = n;
    }
  }

  void get_dims(std::vector<std::vector<size_t> >& dimss__) const {
    dimss__.resize(0);
    std::vector<size_t> scalar_dim;
    dimss__.push_back(scalar_dim);
  }

  void constrained_param_names(std::vector<std::string>& param_names__,
                               bool include_tparams__ = true,
                               bool include_gqs__ = true) const {
    param_names__.push_back("theta");
  }

  void get_param_names(std::vector<std::string>& names) const {
    constrained_param_names(names);
  }

  void unconstrained_param_names(std::vector<std::string>& param_names__,
                                 bool include_tparams__ = true,
                                 bool include_gqs__ = true) const {
    param_names__.clear();
    for (size_t n = 0; n < num_params_r__; ++n) {
      std::stringstream param_name;
      param_name << "param_" << n;
      param_names__.push_back(param_name.str());
    }
  }
  template <typename RNG>
  void write_array(RNG& base_rng__,
                   std::vector<double>& params_r__,
                   std::vector<int>& params_i__,
                   std::vector<double>& vars__,
                   bool include_tparams__ = true,
                   bool include_gqs__ = true,
                   std::ostream* pstream__ = 0) const {
    ++write_array_calls;
    vars__.resize(0);
    for (size_t i = 0; i < params_r__.size(); ++i)
      vars__.push_back(params_r__[i]);
  }

  mutable int templated_log_prob_calls;
  mutable int transform_inits_calls;
  mutable int write_array_calls;
  double log_prob_return_value;
};

}

TEST_F(ServicesUtilInitialize, model_throws__radius_zero) {
  test::mock_throwing_model throwing_model;

  double init_radius = 0;
  bool print_timing = false;
  EXPECT_THROW(stan::services::util::initialize(throwing_model, empty_context, rng,
                                                init_radius, print_timing,
                                                logger, init),
               std::domain_error);

  EXPECT_EQ(3, logger.call_count());
  EXPECT_EQ(3, logger.call_count_info());
  EXPECT_EQ(1, logger.find_info("throwing within log_prob"));
}

TEST_F(ServicesUtilInitialize, model_throws__radius_two) {
  test::mock_throwing_model throwing_model;

  double init_radius = 2;
  bool print_timing = false;
  EXPECT_THROW(stan::services::util::initialize(throwing_model, empty_context, rng,
                                                init_radius, print_timing,
                                                logger, init),
               std::domain_error);
  EXPECT_EQ(303, logger.call_count());
  EXPECT_EQ(303, logger.call_count_info());
  EXPECT_EQ(100, logger.find_info("throwing within log_prob"));
}

TEST_F(ServicesUtilInitialize, model_throws__full_init) {
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

  test::mock_throwing_model throwing_model;

  double init_radius = 2;
  bool print_timing = false;
  EXPECT_THROW(stan::services::util::initialize(throwing_model, init_context, rng,
                                                init_radius, print_timing,
                                                logger, init),
               std::domain_error);
  EXPECT_EQ(303, logger.call_count());
  EXPECT_EQ(303, logger.call_count_info());
  EXPECT_EQ(100, logger.find_info("throwing within log_prob"));
}


namespace test {
// Mock Throwing Model throws exception
class mock_error_model: public stan::model::prob_grad {
 public:

  mock_error_model():
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

  template <bool propto__, bool jacobian__, typename T__>
  T__ log_prob(std::vector<T__>& params_r__,
               std::vector<int>& params_i__,
               std::ostream* pstream__ = 0) const {
    ++templated_log_prob_calls;
    throw std::out_of_range("out_of_range error in log_prob");
    return log_prob_return_value;
  }

  void transform_inits(const stan::io::var_context& context__,
                       std::vector<int>& params_i__,
                       std::vector<double>& params_r__,
                       std::ostream* pstream__) const {
    ++transform_inits_calls;
    for (size_t n = 0; n < params_r__.size(); ++n) {
      params_r__[n] = n;
    }
  }

  void get_dims(std::vector<std::vector<size_t> >& dimss__) const {
    dimss__.resize(0);
    std::vector<size_t> scalar_dim;
    dimss__.push_back(scalar_dim);
  }

  void constrained_param_names(std::vector<std::string>& param_names__,
                               bool include_tparams__ = true,
                               bool include_gqs__ = true) const {
    param_names__.push_back("theta");
  }

  void get_param_names(std::vector<std::string>& names) const {
    constrained_param_names(names);
  }

  void unconstrained_param_names(std::vector<std::string>& param_names__,
                                 bool include_tparams__ = true,
                                 bool include_gqs__ = true) const {
    param_names__.clear();
    for (size_t n = 0; n < num_params_r__; ++n) {
      std::stringstream param_name;
      param_name << "param_" << n;
      param_names__.push_back(param_name.str());
    }
  }
  template <typename RNG>
  void write_array(RNG& base_rng__,
                   std::vector<double>& params_r__,
                   std::vector<int>& params_i__,
                   std::vector<double>& vars__,
                   bool include_tparams__ = true,
                   bool include_gqs__ = true,
                   std::ostream* pstream__ = 0) const {
    ++write_array_calls;
    vars__.resize(0);
    for (size_t i = 0; i < params_r__.size(); ++i)
      vars__.push_back(params_r__[i]);
  }

  mutable int templated_log_prob_calls;
  mutable int transform_inits_calls;
  mutable int write_array_calls;
  double log_prob_return_value;
};
}


TEST_F(ServicesUtilInitialize, model_errors__radius_zero) {
  test::mock_error_model error_model;

  double init_radius = 0;
  bool print_timing = false;
  EXPECT_THROW_MSG(stan::services::util::initialize(error_model, empty_context, rng,
                                                    init_radius, print_timing,
                                                    logger, init),
                   std::out_of_range,
                   "out_of_range error in log_prob");
  EXPECT_EQ(2, logger.call_count());
  EXPECT_EQ(2, logger.call_count_info());
  EXPECT_EQ(1, logger.find_info("out_of_range error in log_prob"));
  EXPECT_EQ(1, logger.find_info("Unrecoverable error evaluating the log probability at the initial value."));
}

TEST_F(ServicesUtilInitialize, model_errors__radius_two) {
  test::mock_error_model error_model;

  double init_radius = 2;
  bool print_timing = false;
  EXPECT_THROW_MSG(stan::services::util::initialize(error_model, empty_context, rng,
                                                    init_radius, print_timing,
                                                    logger, init),
                   std::out_of_range,
                   "out_of_range error in log_prob");
  EXPECT_EQ(2, logger.call_count());
  EXPECT_EQ(2, logger.call_count_info());
  EXPECT_EQ(1, logger.find_info("out_of_range error in log_prob"));
}

TEST_F(ServicesUtilInitialize, model_errors__full_init) {
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

  test::mock_error_model error_model;

  double init_radius = 2;
  bool print_timing = false;
  EXPECT_THROW_MSG(stan::services::util::initialize(error_model, init_context, rng,
                                                    init_radius, print_timing,
                                                    logger, init),
                   std::out_of_range,
                   "out_of_range error in log_prob");
  EXPECT_EQ(2, logger.call_count());
  EXPECT_EQ(2, logger.call_count_info());
  EXPECT_EQ(1, logger.find_info("out_of_range error in log_prob"));
}

namespace test {
// mock_throwing_model_in_write_array throws exception in the write_array()
// method
class mock_throwing_model_in_write_array: public stan::model::prob_grad {
 public:

  mock_throwing_model_in_write_array():
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

  template <bool propto__, bool jacobian__, typename T__>
  T__ log_prob(std::vector<T__>& params_r__,
               std::vector<int>& params_i__,
               std::ostream* pstream__ = 0) const {
    ++templated_log_prob_calls;
    return log_prob_return_value;
  }

  void transform_inits(const stan::io::var_context& context__,
                       std::vector<int>& params_i__,
                       std::vector<double>& params_r__,
                       std::ostream* pstream__) const {
    ++transform_inits_calls;
    for (size_t n = 0; n < params_r__.size(); ++n) {
      params_r__[n] = n;
    }
  }

  void get_dims(std::vector<std::vector<size_t> >& dimss__) const {
    dimss__.resize(0);
    std::vector<size_t> scalar_dim;
    dimss__.push_back(scalar_dim);
  }

  void constrained_param_names(std::vector<std::string>& param_names__,
                               bool include_tparams__ = true,
                               bool include_gqs__ = true) const {
    param_names__.push_back("theta");
  }

  void get_param_names(std::vector<std::string>& names) const {
    constrained_param_names(names);
  }

  void unconstrained_param_names(std::vector<std::string>& param_names__,
                                 bool include_tparams__ = true,
                                 bool include_gqs__ = true) const {
    param_names__.clear();
    for (size_t n = 0; n < num_params_r__; ++n) {
      std::stringstream param_name;
      param_name << "param_" << n;
      param_names__.push_back(param_name.str());
    }
  }
  template <typename RNG>
  void write_array(RNG& base_rng__,
                   std::vector<double>& params_r__,
                   std::vector<int>& params_i__,
                   std::vector<double>& vars__,
                   bool include_tparams__ = true,
                   bool include_gqs__ = true,
                   std::ostream* pstream__ = 0) const {
    ++write_array_calls;
    throw std::domain_error("throwing within write_array");
    vars__.resize(0);
    for (size_t i = 0; i < params_r__.size(); ++i)
      vars__.push_back(params_r__[i]);
  }

  mutable int templated_log_prob_calls;
  mutable int transform_inits_calls;
  mutable int write_array_calls;
  double log_prob_return_value;
};
}

TEST_F(ServicesUtilInitialize, model_throws_in_write_array__radius_zero) {
  test::mock_throwing_model_in_write_array throwing_model;

  double init_radius = 0;
  bool print_timing = false;
  EXPECT_THROW(stan::services::util::initialize(throwing_model, empty_context, rng,
                                                init_radius, print_timing,
                                                logger, init),
               std::domain_error);

  EXPECT_EQ(3, logger.call_count());
  EXPECT_EQ(3, logger.call_count_info());
  EXPECT_EQ(1, logger.find_info("throwing within write_array"));
}

TEST_F(ServicesUtilInitialize, model_throws_in_write_array__radius_two) {
  test::mock_throwing_model_in_write_array throwing_model;

  double init_radius = 2;
  bool print_timing = false;
  EXPECT_THROW(stan::services::util::initialize(throwing_model, empty_context, rng,
                                                init_radius, print_timing,
                                                logger, init),
               std::domain_error);
  EXPECT_EQ(303, logger.call_count());
  EXPECT_EQ(303, logger.call_count_info());
  EXPECT_EQ(100, logger.find_info("throwing within write_array"));
}

TEST_F(ServicesUtilInitialize, model_throws_in_write_array__full_init) {
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

  test::mock_throwing_model_in_write_array throwing_model;

  double init_radius = 2;
  bool print_timing = false;
  EXPECT_THROW(stan::services::util::initialize(throwing_model, init_context, rng,
                                                init_radius, print_timing,
                                                logger, init),
               std::domain_error);
  EXPECT_EQ(303, logger.call_count());
  EXPECT_EQ(303, logger.call_count_info());
  EXPECT_EQ(100, logger.find_info("throwing within write_array"));
}
