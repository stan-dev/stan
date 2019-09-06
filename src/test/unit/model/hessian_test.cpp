#include <stan/model/hessian.hpp>
#include <test/test-models/good/model/valid.hpp>
#include <gtest/gtest.h>

TEST(ModelUtil, hessian) {
  int dim = 5;
  
  Eigen::VectorXd x(dim);
  double f;
  Eigen::VectorXd grad_f(dim);
  Eigen::MatrixXd hess_f(dim, dim);
  
  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  std::stringstream output;
  valid_model_namespace::valid_model valid_model(data_var_context, &output);
  EXPECT_NO_THROW(stan::model::hessian(valid_model, x, f, grad_f, hess_f));
  
  EXPECT_FLOAT_EQ(dim, x.size());
  EXPECT_FLOAT_EQ(dim, grad_f.size());
  EXPECT_FLOAT_EQ(dim, hess_f.rows());
  EXPECT_FLOAT_EQ(dim, hess_f.cols());
  
  EXPECT_EQ("", output.str());

  // Incorporate once operands and partials has been generalized
  //output.str("");
  //domain_fail_namespace::domain_fail domain_fail_model(data_var_context, &output);
  //EXPECT_THROW(stan::model::hessian(domain_fail_model, x, f, grad_f, hess_f), std::domain_error);
  //EXPECT_EQ("", output.str());
}
