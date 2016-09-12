#include <stan/model/grad_hess_log_prob.hpp>
#include <test/test-models/good/model/valid.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

TEST(ModelUtil, grad_hess_log_prob) {
  stan::test::capture_std_streams();

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  stan_model model(data_var_context, 0);
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;

  std::stringstream out;
  
  try {
    std::vector<double> hessian;
    stan::model::grad_hess_log_prob<true, true, stan_model>(model, params_r, params_i, gradient, hessian, 0);
    stan::model::grad_hess_log_prob<true, false, stan_model>(model, params_r, params_i, gradient, hessian, 0);
    stan::model::grad_hess_log_prob<false, true, stan_model>(model, params_r, params_i, gradient, hessian, 0);
    stan::model::grad_hess_log_prob<false, false, stan_model>(model, params_r, params_i, gradient, hessian, 0);

    out.str("");
    stan::model::grad_hess_log_prob<true, true, stan_model>(model, params_r, params_i, gradient, hessian, &out);
    stan::model::grad_hess_log_prob<true, false, stan_model>(model, params_r, params_i, gradient, hessian, &out);
    stan::model::grad_hess_log_prob<false, true, stan_model>(model, params_r, params_i, gradient, hessian, &out);
    stan::model::grad_hess_log_prob<false, false, stan_model>(model, params_r, params_i, gradient, hessian, &out);
    EXPECT_EQ("", out.str());
  } catch (...) {
    FAIL() << "grad_hess_log_prob";
  }

  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}
