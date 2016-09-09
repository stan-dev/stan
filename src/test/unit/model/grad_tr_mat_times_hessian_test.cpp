#include <stan/model/grad_tr_mat_times_hessian.hpp>
#include <test/test-models/good/model/valid.hpp>
#include <gtest/gtest.h>

TEST(ModelUtil, grad_tr_mat_times_hessian) {
  int dim = 5;
  
  Eigen::VectorXd x(dim);
  Eigen::MatrixXd X = Eigen::MatrixXd::Identity(dim, dim);
  Eigen::VectorXd grad_tr_X_hess_f(dim);
  
  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  std::stringstream output;

  valid_model_namespace::valid_model valid_model(data_var_context, &output);
  EXPECT_NO_THROW(stan::model::grad_tr_mat_times_hessian(valid_model, x, X, grad_tr_X_hess_f));
  
  EXPECT_FLOAT_EQ(dim, x.size());
  EXPECT_FLOAT_EQ(dim, X.rows());
  EXPECT_FLOAT_EQ(dim, X.cols());
  EXPECT_FLOAT_EQ(dim, grad_tr_X_hess_f.size());

  EXPECT_EQ("", output.str());
  
  // Incorporate once operands and partials has been generalized
  //output.str("");
  //domain_fail_namespace::domain_fail domain_fail_model(data_var_context, &output);
  //EXPECT_THROW(stan::model::grad_tr_mat_times_hessian(domain_fail_model, x, X, grad_tr_X_hess_f),
  //             std::domain_error);
  //EXPECT_EQ("", output.str());
}
