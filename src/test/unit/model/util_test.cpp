#include <gtest/gtest.h>

#include <stan/model/util.hpp>

#include <vector>
#include <stan/io/reader.hpp>
#include <stan/math/matrix/accumulator.hpp>
#include <stan/prob/distributions/univariate/continuous/uniform.hpp>
#include <stan/prob/transform.hpp>

#include <stan/io/dump.hpp>
#include <test/test-models/good/model/valid.cpp>
//#include <test/test-models/good/model/domain_fail.cpp>

class TestModel_uniform_01 {
public:
  template <bool propto__, bool jacobian__, typename T__>
  T__ log_prob(std::vector<T__>& params_r__,
               std::vector<int>& params_i__,
               std::ostream* pstream__ = 0) const {
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    
    // model parameters
    stan::io::reader<T__> in__(params_r__,params_i__);
    
    T__ y;
    if (jacobian__)
      y = in__.scalar_lub_constrain(0,1,lp__);
    else
      y = in__.scalar_lub_constrain(0,1);
    
    lp_accum__.add(stan::prob::uniform_log<propto__>(y, 0, 1));
    lp_accum__.add(lp__);

    return lp_accum__.sum();
  }
};

TEST(ModelUtil, finite_diff_grad__false_false) {
  TestModel_uniform_01 model;
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;
  
  for (int i = 0; i < 10; i++) {
    params_r[0] = (i-5.0) * 10;
    
    stan::model::finite_diff_grad<false,false,TestModel_uniform_01>
      (model, params_r, params_i, gradient);
    
    ASSERT_EQ(1U, gradient.size());
    EXPECT_FLOAT_EQ(0.0, gradient[0]);
  }
}
TEST(ModelUtil, finite_diff_grad__false_true) {
  TestModel_uniform_01 model;
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;
  
  for (int i = 0; i < 10; i++) {
    double x = (i - 5.0) * 10;
    params_r[0] = x;

    stan::model::finite_diff_grad<false,true,TestModel_uniform_01>
      (model, params_r, params_i, gradient);
    
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
  
  for (int i = 0; i < 10; i++) {
    double x = (i - 5.0) * 10;
    params_r[0] = x;

    stan::model::finite_diff_grad<true,false,TestModel_uniform_01>
      (model, params_r, params_i, gradient);
    
    ASSERT_EQ(1U, gradient.size());
    
    EXPECT_FLOAT_EQ(0.0, gradient[0]);
  }
}

TEST(ModelUtil, finite_diff_grad__true_true) {
  TestModel_uniform_01 model;
  std::vector<double> params_r(1);
  std::vector<int> params_i(0);
  std::vector<double> gradient;
  
  for (int i = 0; i < 10; i++) {
    double x = (i - 5.0) * 10;
    params_r[0] = x;

    stan::model::finite_diff_grad<true,true,TestModel_uniform_01>
      (model, params_r, params_i, gradient);
    
    ASSERT_EQ(1U, gradient.size());
    
    double expected_gradient = -std::tanh(0.5 * x);    
    EXPECT_FLOAT_EQ(expected_gradient, gradient[0]);
  }
}

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

TEST(ModelUtil, gradient_dot_vector) {
  
  int dim = 5;
  
  Eigen::VectorXd x(dim);
  Eigen::VectorXd v(dim);
  double f;
  double grad_f_dot_v(dim);
  
  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  

  std::stringstream output;
  valid_model_namespace::valid_model valid_model(data_var_context, &output);
  EXPECT_NO_THROW(stan::model::gradient_dot_vector(valid_model, x, v, f, grad_f_dot_v));
  
  EXPECT_FLOAT_EQ(dim, x.size());
  EXPECT_FLOAT_EQ(dim, v.size());

  EXPECT_EQ("", output.str());
  
  // Incorporate once operands and partials has been generalized
  //output.str("");
  //domain_fail_namespace::domain_fail domain_fail_model(data_var_context, &output);
  //EXPECT_THROW(stan::model::gradient_dot_vector(domain_fail_model, x, v, f, grad_f_dot_v),
  //             std::domain_error);
  //EXPECT_EQ("", output.str());
}

TEST(ModelUtil, hessian_times_vector) {
  int dim = 5;
  
  Eigen::VectorXd x(dim);
  Eigen::VectorXd v(dim);
  double f;
  Eigen::VectorXd hess_f_dot_v(dim);
  
  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  std::stringstream output;
  valid_model_namespace::valid_model valid_model(data_var_context, &output);
  EXPECT_NO_THROW(stan::model::hessian_times_vector(valid_model, x, v, f, hess_f_dot_v));
  
  EXPECT_FLOAT_EQ(dim, x.size());
  EXPECT_FLOAT_EQ(dim, v.size());
  EXPECT_FLOAT_EQ(dim, hess_f_dot_v.size());
  
  EXPECT_EQ("", output.str());

  // Incorporate once operands and partials has been generalized
  //output.str("");
  //domain_fail_namespace::domain_fail domain_fail_model(data_var_context, &output);
  //EXPECT_THROW(stan::model::hessian_times_vector(domain_fail_model, x, v, f, hess_f_dot_v),
  //             std::domain_error);
  //EXPECT_EQ("", output.str());
}

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
