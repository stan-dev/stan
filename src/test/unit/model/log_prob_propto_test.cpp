#include <stan/model/log_prob_propto.hpp>
#include <test/test-models/good/model/valid.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

TEST(ModelUtil, streams) {
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
    stan::model::log_prob_propto<true, stan_model>(model, params_r, params_i, 0);
    stan::model::log_prob_propto<false, stan_model>(model, params_r, params_i, 0);
    out.str("");
    stan::model::log_prob_propto<true, stan_model>(model, params_r, params_i, &out);
    stan::model::log_prob_propto<false, stan_model>(model, params_r, params_i, &out);
    EXPECT_EQ("", out.str());
  } catch (...) {
    FAIL() << "log_prob_propto";
  }


  try {
    Eigen::VectorXd p(1);
    stan::model::log_prob_propto<true, stan_model>(model, p, 0);
    stan::model::log_prob_propto<false, stan_model>(model, p, 0);
    out.str("");
    stan::model::log_prob_propto<true, stan_model>(model, p, &out);
    stan::model::log_prob_propto<false, stan_model>(model, p, &out);
    EXPECT_EQ("", out.str());
  } catch (...) {
    FAIL() << "log_prob_propto";
  }

  
  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}

