#include <gtest/gtest.h>
#include <test/test-models/good/vb/multivariate_with_constraint.hpp>

class call_log_prob : public ::testing::Test {
public:
  void SetUp() {
    const std::string DATA = "";
    std::stringstream data_stream(DATA);
    stan::io::dump dummy_context(data_stream);

    // Instantiate model
    model_ = new stan_model(dummy_context);
    z_tilde1.resize(2);
    z_tilde1[0] = 0;
    z_tilde1[1] = 0;

    z_tilde2.resize(2);
    z_tilde2[0] = 2;
    z_tilde2[1] = 3;

  }

  void TearDown() {
    stan::agrad::recover_memory();
    delete model_;
  }

  stan_model *model_;
  Eigen::Matrix<double, Eigen::Dynamic, 1> z_tilde1;
  Eigen::Matrix<double, Eigen::Dynamic, 1> z_tilde2;
};

// need model_
TEST_F(call_log_prob, with_vars_propto) {
  Eigen::Matrix<stan::agrad::var, Eigen::Dynamic, 1> z_tilde1_var(2);
  z_tilde1_var[0] = z_tilde1[0];
  z_tilde1_var[1] = z_tilde1[1];
  std::cout << "val: " << model_->log_prob<true,true>(z_tilde1_var, &std::cout).val()
            << std::endl;

  Eigen::Matrix<stan::agrad::var, Eigen::Dynamic, 1> z_tilde2_var(2);
  z_tilde2_var[0] = z_tilde2[0];
  z_tilde2_var[1] = z_tilde2[1];
  std::cout << "val: " << model_->log_prob<true,true>(z_tilde2_var, &std::cout).val()
            << std::endl;

}

TEST_F(call_log_prob, with_vars_no_propto) {
  Eigen::Matrix<stan::agrad::var, Eigen::Dynamic, 1> z_tilde1_var(2);
  z_tilde1_var[0] = z_tilde1[0];
  z_tilde1_var[1] = z_tilde1[1];
  std::cout << "val: " << model_->log_prob<false,true>(z_tilde1_var, &std::cout).val()
            << std::endl;

  Eigen::Matrix<stan::agrad::var, Eigen::Dynamic, 1> z_tilde2_var(2);
  z_tilde2_var[0] = z_tilde2[0];
  z_tilde2_var[1] = z_tilde2[1];
  std::cout << "val: " << model_->log_prob<false,true>(z_tilde2_var, &std::cout).val()
            << std::endl;
}

TEST_F(call_log_prob, doubles_propto) {
  std::cout << "out: " << model_->log_prob<true,true>(z_tilde1, &std::cout)
            << std::endl;
  std::cout << "out: " << model_->log_prob<true,true>(z_tilde2, &std::cout)
            << std::endl;
}

TEST_F(call_log_prob, doubles_no_propto) {
  std::cout << "out: " << model_->log_prob<false,true>(z_tilde1, &std::cout)
            << std::endl;
  std::cout << "out: " << model_->log_prob<false,true>(z_tilde2, &std::cout)
            << std::endl;
}
