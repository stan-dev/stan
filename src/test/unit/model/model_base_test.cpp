#include <gtest/gtest.h>
#include <stan/model/model_base.hpp>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct mock_model : public stan::model::model_base {
  mock_model(size_t n) : model_base(n) {}

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

  double log_prob(Eigen::VectorXd& params_r,
                  std::ostream* msgs) const override {
    return 1;
  }

  stan::math::var log_prob(Eigen::Matrix<stan::math::var, -1, 1>& params_r,
                           std::ostream* msgs) const override {
    return 2;
  }

  double log_prob_jacobian(Eigen::VectorXd& params_r,
                           std::ostream* msgs) const override {
    return 3;
  }

  stan::math::var log_prob_jacobian(
      Eigen::Matrix<stan::math::var, -1, 1>& params_r,
      std::ostream* msgs) const override {
    return 4;
  }

  double log_prob_propto(Eigen::VectorXd& params_r,
                         std::ostream* msgs) const override {
    return 5;
  }

  stan::math::var log_prob_propto(
      Eigen::Matrix<stan::math::var, -1, 1>& params_r,
      std::ostream* msgs) const override {
    return 6;
  }

  double log_prob_propto_jacobian(Eigen::VectorXd& params_r,
                                  std::ostream* msgs) const override {
    return 7;
  }

  stan::math::var log_prob_propto_jacobian(
      Eigen::Matrix<stan::math::var, -1, 1>& params_r,
      std::ostream* msgs) const override {
    return 8;
  }

  void transform_inits(const stan::io::var_context& context,
                       Eigen::VectorXd& params_r,
                       std::ostream* msgs) const override {}

  void write_array(boost::ecuyer1988& base_rng, Eigen::VectorXd& params_r,
                   Eigen::VectorXd& params_constrained_r, bool include_tparams,
                   bool include_gqs, std::ostream* msgs) const override {}

  double log_prob(std::vector<double>& params_r, std::vector<int>& params_i,
                  std::ostream* msgs) const override {
    return 11;
  }

  stan::math::var log_prob(std::vector<stan::math::var>& params_r,
                           std::vector<int>& params_i,
                           std::ostream* msgs) const override {
    return 12;
  }

  double log_prob_jacobian(std::vector<double>& params_r,
                           std::vector<int>& params_i,
                           std::ostream* msgs) const override {
    return 13;
  }

  stan::math::var log_prob_jacobian(std::vector<stan::math::var>& params_r,
                                    std::vector<int>& params_i,
                                    std::ostream* msgs) const override {
    return 14;
  }

  double log_prob_propto(std::vector<double>& params_r,
                         std::vector<int>& params_i,
                         std::ostream* msgs) const override {
    return 15;
  }

  stan::math::var log_prob_propto(std::vector<stan::math::var>& params_r,
                                  std::vector<int>& params_i,
                                  std::ostream* msgs) const override {
    return 16;
  }

  double log_prob_propto_jacobian(std::vector<double>& params_r,
                                  std::vector<int>& params_i,
                                  std::ostream* msgs) const override {
    return 17;
  }

  stan::math::var log_prob_propto_jacobian(
      std::vector<stan::math::var>& params_r, std::vector<int>& params_i,
      std::ostream* msgs) const override {
    return 0;
  }

  void transform_inits(const stan::io::var_context& context,
                       std::vector<int>& params_i,
                       std::vector<double>& params_r,
                       std::ostream* msgs) const override {}

  void write_array(boost::ecuyer1988& base_rng, std::vector<double>& params_r,
                   std::vector<int>& params_i,
                   std::vector<double>& params_r_constrained,
                   bool include_tparams, bool include_gqs,
                   std::ostream* msgs) const override {}
};

TEST(model, modelBaseInheritance) {
  // check that prob_grad inheritance works
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

  // these versions defined in mock_model; make sure they work from base
  EXPECT_FLOAT_EQ(1, bm.log_prob(params_r, msgs));
  EXPECT_FLOAT_EQ(2, bm.log_prob(params_r_v, msgs).val());
  EXPECT_FLOAT_EQ(3, bm.log_prob_jacobian(params_r, msgs));
  EXPECT_FLOAT_EQ(4, bm.log_prob_jacobian(params_r_v, msgs).val());
  EXPECT_FLOAT_EQ(5, bm.log_prob_propto(params_r, msgs));
  EXPECT_FLOAT_EQ(6, bm.log_prob_propto(params_r_v, msgs).val());
  EXPECT_FLOAT_EQ(7, bm.log_prob_propto_jacobian(params_r, msgs));
  EXPECT_FLOAT_EQ(8, bm.log_prob_propto_jacobian(params_r_v, msgs).val());

  // test template version from base class;  not callable from mock_model
  // because templated class functions are not inherited
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
