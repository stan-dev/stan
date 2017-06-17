#include <stan/model/gradient.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/test-models/good/model/valid.hpp>
#include <gtest/gtest.h>

TEST(ModelUtil, gradient) {
  int dim = 5;

  Eigen::VectorXd x(dim);
  double f;
  Eigen::VectorXd g(dim);

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream output;
  valid_model_namespace::valid_model valid_model(data_var_context, &output);
  EXPECT_NO_THROW(stan::model::gradient(valid_model, x, f, g));

  EXPECT_FLOAT_EQ(dim, x.size());
  EXPECT_FLOAT_EQ(dim, g.size());

  EXPECT_EQ("", output.str());

  // Incorporate once operands and partials has been generalized
  //output.str("");
  //domain_fail_namespace::domain_fail domain_fail_model(data_var_context, &output);
  //EXPECT_THROW(stan::model::gradient(domain_fail_model, x, f, g), std::domain_error);
  //EXPECT_EQ("", output.str());
}

TEST(ModelUtil, gradient_writer) {
  int dim = 5;

  Eigen::VectorXd x(dim);
  double f;
  Eigen::VectorXd g(dim);

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream output;
  stan::test::unit::instrumented_logger logger;
  valid_model_namespace::valid_model valid_model(data_var_context, &output);
  EXPECT_NO_THROW(stan::model::gradient(valid_model, x, f, g, logger));

  EXPECT_FLOAT_EQ(dim, x.size());
  EXPECT_FLOAT_EQ(dim, g.size());

  EXPECT_EQ(0, logger.call_count());

  // Incorporate once operands and partials has been generalized
  //output.str("");
  //domain_fail_namespace::domain_fail domain_fail_model(data_var_context, &output);
  //EXPECT_THROW(stan::model::gradient(domain_fail_model, x, f, g), std::domain_error);
  //EXPECT_EQ("", output.str());
}
