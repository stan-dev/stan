// The test cannot find stan/math/mix/mat.hpp.
#include <stan/model/hessian.hpp>
#include <test/test-models/good/model/valid.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

TEST(ModelUtil, streams) {
  stan::test::capture_std_streams();

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();


  stan_model model(data_var_context, static_cast<std::stringstream*>(0));
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  Eigen::VectorXd grad_f(dim);
  Eigen::MatrixXd hess_f(dim, dim);

  std::stringstream out;

  try {
    stan::model::log_prob_hessian<true, true, stan_model>(model, params_r, grad_f, hess_f, 0);
    stan::model::log_prob_hessian<true, false, stan_model>(model, params_r, grad_f, hess_f, 0);
    stan::model::log_prob_hessian<false, true, stan_model>(model, params_r, grad_f, hess_f, 0);
    stan::model::log_prob_hessian<false, false, stan_model>(model, params_r, grad_f, hess_f, 0);
    out.str("");
    stan::model::log_prob_hessian<true, true, stan_model>(model, params_r, grad_f, hess_f, &out);
    stan::model::log_prob_hessian<true, false, stan_model>(model, params_r, grad_f, hess_f, &out);
    stan::model::log_prob_hessian<false, true, stan_model>(model, params_r, grad_f, hess_f, &out);
    stan::model::log_prob_hessian<false, false, stan_model>(model, params_r, grad_f, hess_f, &out);
    EXPECT_EQ("", out.str());
  } catch (...) {
    FAIL() << "log_prob_hessian";
  }

  try {
    Eigen::VectorXd p(1);
    Eigen::VectorXd g(1);
    Eigen::MatrixXd h(1, 1);
    stan::model::log_prob_hessian<true, true, stan_model>(model, p, g, h, 0);
    stan::model::log_prob_hessian<true, false, stan_model>(model, p, g, h, 0);
    stan::model::log_prob_hessian<false, true, stan_model>(model, p, g, h, 0);
    stan::model::log_prob_hessian<false, false, stan_model>(model, p, g, h, 0);
    out.str("");
    stan::model::log_prob_hessian<true, true, stan_model>(model, p, g, h, &out);
    stan::model::log_prob_hessian<true, false, stan_model>(model, p, g, h, &out);
    stan::model::log_prob_hessian<false, true, stan_model>(model, p, g, h, &out);
    stan::model::log_prob_hessian<false, false, stan_model>(model, p, g, h, &out);
    EXPECT_EQ("", out.str());
  } catch (...) {
    FAIL() << "log_prob_hessian";
  }

  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}
