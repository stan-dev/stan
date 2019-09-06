#include <stan/model/log_prob_grad.hpp>
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
  std::vector<double> gradient;

  std::stringstream out;

  try {
    stan::model::log_prob_grad<true, true, stan_model>(model, params_r, params_i, gradient, 0);
    stan::model::log_prob_grad<true, false, stan_model>(model, params_r, params_i, gradient, 0);
    stan::model::log_prob_grad<false, true, stan_model>(model, params_r, params_i, gradient, 0);
    stan::model::log_prob_grad<false, false, stan_model>(model, params_r, params_i, gradient, 0);
    out.str("");
    stan::model::log_prob_grad<true, true, stan_model>(model, params_r, params_i, gradient, &out);
    stan::model::log_prob_grad<true, false, stan_model>(model, params_r, params_i, gradient, &out);
    stan::model::log_prob_grad<false, true, stan_model>(model, params_r, params_i, gradient, &out);
    stan::model::log_prob_grad<false, false, stan_model>(model, params_r, params_i, gradient, &out);
    EXPECT_EQ("", out.str());
  } catch (...) {
    FAIL() << "log_prob_grad";
  }

  try {
    Eigen::VectorXd p(1);
    Eigen::VectorXd g(1);
    stan::model::log_prob_grad<true, true, stan_model>(model, p, g, 0);
    stan::model::log_prob_grad<true, false, stan_model>(model, p, g, 0);
    stan::model::log_prob_grad<false, true, stan_model>(model, p, g, 0);
    stan::model::log_prob_grad<false, false, stan_model>(model, p, g, 0);
    out.str("");
    stan::model::log_prob_grad<true, true, stan_model>(model, p, g, &out);
    stan::model::log_prob_grad<true, false, stan_model>(model, p, g, &out);
    stan::model::log_prob_grad<false, true, stan_model>(model, p, g, &out);
    stan::model::log_prob_grad<false, false, stan_model>(model, p, g, &out);
    EXPECT_EQ("", out.str());
  } catch (...) {
    FAIL() << "log_prob_grad";
  }

  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}
