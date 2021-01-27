#include <gtest/gtest.h>
#include <stan/model/model_base.hpp>
#include <stan/model/model_base_crtp.hpp>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

struct mock_model : public stan::model::model_base_crtp<mock_model> {
  mock_model(size_t n) : model_base_crtp(n) {}

  virtual ~mock_model() {}

  std::string model_name() const override { return "mock_model"; }

  std::vector<std::string> model_compile_info() const {
    std::vector<std::string> stanc_info;
    stanc_info.push_back("stanc_version = stanc3");
    return stanc_info;
  }

  void get_param_names(std::vector<std::string>& names) const override {}
  void get_dims(std::vector<std::vector<size_t> >& dimss) const override {}

  void constrained_param_names(std::vector<std::string>& param_names,
                               bool include_tparams,
                               bool include_gqs) const override {}

  void unconstrained_param_names(std::vector<std::string>& param_names,
                                 bool include_tparams,
                                 bool include_gqs) const override {}

  template <bool propto, bool jacobian, typename T>
  T log_prob(Eigen::Matrix<T, -1, 1>& params_r, std::ostream* msgs) const {
    if (std::is_same<T, double>::value) {
      if (!propto && !jacobian)
        return 1;
      else if (!propto && jacobian)
        return 3;
      else if (propto && !jacobian)
        return 5;
      else
        return 7;
    } else {
      if (!propto && !jacobian)
        return 2;
      else if (!propto && jacobian)
        return 4;
      else if (propto && !jacobian)
        return 6;
      else
        return 8;
    }
  }

  void transform_inits(const stan::io::var_context& context,
                       Eigen::VectorXd& params_r,
                       std::ostream* msgs) const override {}

  template <typename RNG>
  void write_array(RNG& base_rng, Eigen::VectorXd& params_r,
                   Eigen::VectorXd& params_constrained_r, bool include_tparams,
                   bool include_gqs, std::ostream* msgs) const {}

  template <bool propto, bool jacobian, typename T>
  T log_prob(std::vector<T>& params_r, std::vector<int>& params_i,
             std::ostream* msgs) const {
    if (std::is_same<T, double>::value) {
      if (!propto && !jacobian)
        return 1;
      else if (!propto && jacobian)
        return 3;
      else if (propto && !jacobian)
        return 5;
      else
        return 7;
    } else {
      if (!propto && !jacobian)
        return 2;
      else if (!propto && jacobian)
        return 4;
      else if (propto && !jacobian)
        return 6;
      else
        return 8;
    }
  }

  void transform_inits(const stan::io::var_context& context,
                       std::vector<int>& params_i,
                       std::vector<double>& params_r,
                       std::ostream* msgs) const override {}

  template <typename RNG>
  void write_array(RNG& base_rng, std::vector<double>& params_r,
                   std::vector<int>& params_i,
                   std::vector<double>& params_r_constrained,
                   bool include_tparams, bool include_gqs,
                   std::ostream* msgs) const {}
};

TEST(model, modelBaseInheritance) {
  // check that base_model and prob_grad inheritance works
  mock_model m(17);
  EXPECT_EQ(17u, m.num_params_r());
  EXPECT_EQ(0u, m.num_params_i());
  EXPECT_THROW(m.param_range_i(0), std::out_of_range);
}

TEST(model, modelTemplateLogProb) {
  mock_model m(17);
  stan::model::model_base& bm = m;
  Eigen::VectorXd params_r(2);
  Eigen::Matrix<stan::math::var, -1, 1> params_r_v(3);
  std::stringstream ss;
  std::ostream* msgs = &ss;

  // test from base class reference
  EXPECT_FLOAT_EQ(1, bm.log_prob(params_r, msgs));
  EXPECT_FLOAT_EQ(2, bm.log_prob(params_r_v, msgs).val());
  EXPECT_FLOAT_EQ(3, bm.log_prob_jacobian(params_r, msgs));
  EXPECT_FLOAT_EQ(4, bm.log_prob_jacobian(params_r_v, msgs).val());
  EXPECT_FLOAT_EQ(5, bm.log_prob_propto(params_r, msgs));
  EXPECT_FLOAT_EQ(6, bm.log_prob_propto(params_r_v, msgs).val());
  EXPECT_FLOAT_EQ(7, bm.log_prob_propto_jacobian(params_r, msgs));
  EXPECT_FLOAT_EQ(8, bm.log_prob_propto_jacobian(params_r_v, msgs).val());

  // test template version from base class reference
  // long form assignment avoids test macro parse error with multi tparams
  double v1 = bm.template log_prob<false, false>(params_r, msgs);
  EXPECT_FLOAT_EQ(1, v1);
  double v2 = bm.template log_prob<false, false>(params_r_v, msgs).val();
  EXPECT_FLOAT_EQ(2, v2);
  double v3 = bm.template log_prob<false, true>(params_r, msgs);
  EXPECT_FLOAT_EQ(3, v3);
  double v4 = bm.template log_prob<false, true>(params_r_v, msgs).val();
  EXPECT_FLOAT_EQ(4, v4);
  double v5 = bm.template log_prob<true, false>(params_r, msgs);
  EXPECT_FLOAT_EQ(5, v5);
  double v6 = bm.template log_prob<true, false>(params_r_v, msgs).val();
  EXPECT_FLOAT_EQ(6, v6);
  double v7 = bm.template log_prob<true, true>(params_r, msgs);
  EXPECT_FLOAT_EQ(7, v7);
  double v8 = bm.template log_prob<true, true>(params_r_v, msgs).val();
  EXPECT_FLOAT_EQ(8, v8);
}
