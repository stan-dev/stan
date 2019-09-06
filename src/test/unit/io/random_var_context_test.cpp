#include <stan/io/random_var_context.hpp>
#include <stan/io/empty_var_context.hpp>
#include <gtest/gtest.h>
#include <boost/random/additive_combine.hpp>  // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>
#include <test/test-models/good/services/test_lp.hpp>
#include <test/unit/util.hpp>

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


class random_var_context : public testing::Test {
 public:
  random_var_context()
      : empty_context(),
        model(empty_context, static_cast<std::stringstream*>(0)),
        rng(0),
        throwing_model() { }

  stan::io::empty_var_context empty_context;
  stan_model model;
  boost::ecuyer1988 rng;
  test::mock_throwing_model_in_write_array throwing_model;  
};

TEST_F(random_var_context, contains_r) {
  stan::io::random_var_context context(model, rng, 2, false);
  EXPECT_FALSE(context.contains_r(""));
  EXPECT_TRUE(context.contains_r("y"));
}

TEST_F(random_var_context, vals_r) {
  stan::io::random_var_context context(model, rng, 2, false);
  std::vector<double> vals_r;
  EXPECT_NO_THROW(vals_r = context.vals_r(""));
  EXPECT_EQ(0, vals_r.size());

  EXPECT_NO_THROW(vals_r = context.vals_r("y"));
  ASSERT_EQ(2, vals_r.size());
  EXPECT_GT(vals_r[0], -10);
  EXPECT_LT(vals_r[0], 10);
  EXPECT_GT(vals_r[1], -100);
  EXPECT_LT(vals_r[1], 10);
}

TEST_F(random_var_context, dims_r) {
  stan::io::random_var_context context(model, rng, 2, false);
  std::vector<size_t> dims_r;
  EXPECT_NO_THROW(dims_r = context.dims_r(""));
  EXPECT_EQ(0, dims_r.size());


  EXPECT_NO_THROW(dims_r = context.dims_r("y"));
  ASSERT_EQ(1, dims_r.size());
  EXPECT_EQ(2, dims_r[0]);
}

TEST_F(random_var_context, contains_i) {
  stan::io::random_var_context context(model, rng, 2, false);
  EXPECT_FALSE(context.contains_i(""));
}

TEST_F(random_var_context, vals_i) {
  stan::io::random_var_context context(model, rng, 2, false);
  std::vector<int> vals_i;
  EXPECT_NO_THROW(vals_i = context.vals_i(""));
  EXPECT_EQ(0, vals_i.size());
}

TEST_F(random_var_context, dims_i) {
  stan::io::random_var_context context(model, rng, 2, false);
  std::vector<size_t> dims_i;
  EXPECT_NO_THROW(dims_i = context.dims_i(""));
  EXPECT_EQ(0, dims_i.size());
}

TEST_F(random_var_context, names_r) {
  stan::io::random_var_context context(model, rng, 2, false);
  std::vector<std::string> names_r;
  EXPECT_NO_THROW(context.names_r(names_r));
  EXPECT_EQ(1, names_r.size());
}

TEST_F(random_var_context, names_i) {
  stan::io::random_var_context context(model, rng, 2, false);
  std::vector<std::string> names_i;
  EXPECT_NO_THROW(context.names_i(names_i));
  EXPECT_EQ(0, names_i.size());
}

TEST_F(random_var_context, construct) {
  EXPECT_THROW_MSG(stan::io::random_var_context(throwing_model, rng, 2, false),
                   std::domain_error,
                   "throwing within write_array");
}
