#include <ostream>
#include <stan/io/var_context.hpp>
#include <stan/io/dump.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/model/prob_grad.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <stan/variational/advi.hpp>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG

typedef boost::ecuyer1988 rng_t;

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
                       Eigen::VectorXd& params_r__,
                       std::ostream* out) const {
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

  void get_param_names(std::vector<std::string>& names) const {
    constrained_param_names(names);
  }

  void unconstrained_param_names(std::vector<std::string>& param_names__,
                                 bool include_tparams__ = true,
                                 bool include_gqs__ = true) const {
    param_names__.clear();
    for (size_t n = 0; n < num_params_r__; n++) {
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

// Mock Throwing Model throws exception
class mock_throwing_model: public stan::model::prob_grad {
public:

  mock_throwing_model(size_t num_params_r):
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
    throw std::domain_error("throwing within log_prob");
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

  void get_param_names(std::vector<std::string>& names) const {
    constrained_param_names(names);
  }

  void unconstrained_param_names(std::vector<std::string>& param_names__,
                                 bool include_tparams__ = true,
                                 bool include_gqs__ = true) const {
    param_names__.clear();
    for (size_t n = 0; n < num_params_r__; n++) {
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


class stochastic_gradient_ascent_test : public testing::Test {
public:
  stochastic_gradient_ascent_test() :
    model(3),
    throwing_model(3),
    writer(output) {}

  void SetUp() {
    cont_params = Eigen::VectorXd::Zero(3);
    model.reset();
    rng.reset();
    output.clear();
  }

  std::string init;
  Eigen::VectorXd cont_params;
  mock_model model;
  mock_throwing_model throwing_model;
  mock_rng rng;
  std::stringstream output;
  stan::callbacks::stream_writer writer;
};

TEST_F(stochastic_gradient_ascent_test, initialize_state_zero_negative_infinity) {
  model.log_prob_return_value =
    -std::numeric_limits<double>::infinity();

  stan::variational::advi<mock_model,
                          stan::variational::normal_meanfield,
                          mock_rng> *advi_meanfield =
    new stan::variational::advi<mock_model,
                                stan::variational::normal_meanfield,
                                mock_rng>
    (model,
     cont_params, rng,
     1, 100,
     100, 1);

  stan::variational::advi<mock_model,
                          stan::variational::normal_fullrank,
                          mock_rng> *advi_fullrank =
    new stan::variational::advi<mock_model,
                                stan::variational::normal_fullrank,
                                mock_rng>
    (model,
     cont_params, rng,
     1, 100,
     100, 1);

  stan::variational::normal_meanfield meanfield_init =
    stan::variational::normal_meanfield(cont_params);
  stan::variational::normal_fullrank fullrank_init =
    stan::variational::normal_fullrank(cont_params);

  std::string error = "stan::variational::advi::calc_ELBO: "
                      "The number of dropped evaluations "
                      "has reached its maximum amount (100). "
                      "Your model may be either severely "
                      "ill-conditioned or misspecified.";

  EXPECT_THROW_MSG(advi_meanfield->stochastic_gradient_ascent(meanfield_init,
                                                              1.0,
                                                              0.01,
                                                              1000,
                                                              writer,
                                                              writer),
                   std::domain_error, error);
  EXPECT_THROW_MSG(advi_fullrank->stochastic_gradient_ascent(fullrank_init,
                                                             1.0,
                                                             0.01,
                                                             1000,
                                                             writer,
                                                             writer),
                   std::domain_error, error);

  delete advi_meanfield;
  delete advi_fullrank;
}

TEST_F(stochastic_gradient_ascent_test, initialize_state_zero_grad_error) {
  throwing_model.log_prob_return_value =
    -std::numeric_limits<double>::infinity();

  stan::variational::advi<mock_throwing_model,
                          stan::variational::normal_meanfield,
                          mock_rng> *advi_meanfield =
    new stan::variational::advi<mock_throwing_model,
                                stan::variational::normal_meanfield,
                                mock_rng>
    (throwing_model,
     cont_params, rng,
     1, 100,
     100, 1);

  stan::variational::advi<mock_throwing_model,
                          stan::variational::normal_fullrank,
                          mock_rng> *advi_fullrank =
    new stan::variational::advi<mock_throwing_model,
                                stan::variational::normal_fullrank,
                                mock_rng>
    (throwing_model,
     cont_params, rng,
     1, 100,
     100, 1);

  stan::variational::normal_meanfield meanfield_init =
    stan::variational::normal_meanfield(cont_params);
  stan::variational::normal_fullrank fullrank_init =
    stan::variational::normal_fullrank(cont_params);

  std::string error = "stan::variational::normal_meanfield::calc_grad: "
                      "The number of dropped evaluations "
                      "has reached its maximum amount (10). "
                      "Your model may be either severely "
                      "ill-conditioned or misspecified.";

  EXPECT_THROW_MSG(advi_meanfield->stochastic_gradient_ascent(meanfield_init,
                                                              1.0,
                                                              0.01,
                                                              1000,
                                                              writer,
                                                              writer),
                   std::domain_error, error);

  error = "stan::variational::normal_fullrank::calc_grad: "
                      "The number of dropped evaluations "
                      "has reached its maximum amount (10). "
                      "Your model may be either severely "
                      "ill-conditioned or misspecified.";

  EXPECT_THROW_MSG(advi_fullrank->stochastic_gradient_ascent(fullrank_init,
                                                             1.0,
                                                             0.01,
                                                             1000,
                                                             writer,
                                                             writer),
                   std::domain_error, error);

  delete advi_meanfield;
  delete advi_fullrank;
}



