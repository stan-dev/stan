#include <stan/model/finite_diff_grad.hpp>
#include <test/unit/model/test_model.hpp>
#include <test/test-models/good/model/valid.hpp>
#include <test/unit/util.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <gtest/gtest.h>

TEST(ModelUtil, finite_diff_grad__false_false) {
  TestModel_uniform_01 model;
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;
  stan::callbacks::interrupt interrupt;

  for (int i = 0; i < 10; i++) {
    params_r[0] = (i-5.0) * 10;

    stan::model::finite_diff_grad<false,false,TestModel_uniform_01>
      (model, interrupt, params_r, params_i, gradient);

    ASSERT_EQ(1U, gradient.size());
    EXPECT_FLOAT_EQ(0.0, gradient[0]);
  }
}
TEST(ModelUtil, finite_diff_grad__false_true) {
  TestModel_uniform_01 model;
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;
  stan::callbacks::interrupt interrupt;

  for (int i = 0; i < 10; i++) {
    double x = (i - 5.0) * 10;
    params_r[0] = x;

    stan::model::finite_diff_grad<false,true,TestModel_uniform_01>
      (model, interrupt, params_r, params_i, gradient);

    ASSERT_EQ(1U, gradient.size());

    // derivative of the transform
    double expected_gradient = -std::tanh(0.5 * x);
    EXPECT_FLOAT_EQ(expected_gradient, gradient[0]);
  }
}

TEST(ModelUtil, finite_diff_grad__true_false) {
  TestModel_uniform_01 model;
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;
  stan::callbacks::interrupt interrupt;

  for (int i = 0; i < 10; i++) {
    double x = (i - 5.0) * 10;
    params_r[0] = x;

    stan::model::finite_diff_grad<true,false,TestModel_uniform_01>
      (model, interrupt, params_r, params_i, gradient);

    ASSERT_EQ(1U, gradient.size());

    EXPECT_FLOAT_EQ(0.0, gradient[0]);
  }
}

TEST(ModelUtil, finite_diff_grad__true_true) {
  TestModel_uniform_01 model;
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;
  stan::callbacks::interrupt interrupt;

  for (int i = 0; i < 10; i++) {
    double x = (i - 5.0) * 10;
    params_r[0] = x;

    stan::model::finite_diff_grad<true,true,TestModel_uniform_01>
      (model, interrupt, params_r, params_i, gradient);

    ASSERT_EQ(1U, gradient.size());

    double expected_gradient = -std::tanh(0.5 * x);
    EXPECT_FLOAT_EQ(expected_gradient, gradient[0]);
  }
}

TEST(ModelUtil, streams) {
  stan::test::capture_std_streams();

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  stan_model model(data_var_context, 0);
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;
  stan::callbacks::interrupt interrupt;

  std::stringstream out;

  try {
    stan::model::finite_diff_grad<true, true, stan_model>(model, interrupt, params_r, params_i, gradient, 1e-6, 0);
    stan::model::finite_diff_grad<true, false, stan_model>(model, interrupt, params_r, params_i, gradient, 1e-6, 0);
    stan::model::finite_diff_grad<false, true, stan_model>(model, interrupt, params_r, params_i, gradient, 1e-6, 0);
    stan::model::finite_diff_grad<false, false, stan_model>(model, interrupt, params_r, params_i, gradient, 1e-6, 0);


    out.str("");
    stan::model::finite_diff_grad<true, true, stan_model>(model, interrupt, params_r, params_i, gradient, 1e-6, &out);
    stan::model::finite_diff_grad<true, false, stan_model>(model, interrupt, params_r, params_i, gradient, 1e-6, &out);
    stan::model::finite_diff_grad<false, true, stan_model>(model, interrupt, params_r, params_i, gradient, 1e-6, &out);
    stan::model::finite_diff_grad<false, false, stan_model>(model, interrupt, params_r, params_i, gradient, 1e-6, &out);
    EXPECT_EQ("", out.str());
  } catch (...) {
    FAIL() << "finite_diff_grad";
  }

  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}
