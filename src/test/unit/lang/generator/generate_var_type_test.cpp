#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <test/unit/lang/utility.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>

TEST(langGenerator, genRealVars) {
  using stan::lang::scope;
  using stan::lang::transformed_data_origin;
  using stan::lang::function_argument_origin;
  scope td_origin = transformed_data_origin;
  scope fun_origin = function_argument_origin;
  std::stringstream o;

  o.str(std::string());
  stan::lang::generate_real_var_type(td_origin, true, o);
  EXPECT_EQ(1, count_matches("local_scalar_t__", o.str()));

  o.str(std::string());
  stan::lang::generate_real_var_type(td_origin, false, o);
  EXPECT_EQ(1, count_matches("double", o.str()));

  o.str(std::string());
  stan::lang::generate_real_var_type(fun_origin, true, o);
  EXPECT_EQ(1, count_matches("local_scalar_t__", o.str()));

  o.str(std::string());
  stan::lang::generate_real_var_type(fun_origin, false, o);
  EXPECT_EQ(1, count_matches("local_scalar_t__", o.str()));
}

TEST(langGenerator, genArrayVars) {
  using stan::lang::bare_expr_type;
  using stan::lang::int_type;
  using stan::lang::double_type;
  using stan::lang::vector_type;
  using stan::lang::row_vector_type;
  using stan::lang::matrix_type;
  using stan::lang::scope;
  using stan::lang::transformed_data_origin;
  using stan::lang::function_argument_origin;
  scope td_origin = transformed_data_origin;
  scope fun_origin = function_argument_origin;
  std::stringstream ssReal;
  std::stringstream o;

  ssReal.str(std::string());
  stan::lang::generate_real_var_type(td_origin, true, ssReal);
  o.str(std::string());
  stan::lang::generate_bare_type(bare_expr_type(double_type()),ssReal.str(),o);
  EXPECT_EQ(1, count_matches("local_scalar_t__", o.str()));

  ssReal.str(std::string());
  stan::lang::generate_real_var_type(td_origin, false, ssReal);
  o.str(std::string());
  stan::lang::generate_bare_type(bare_expr_type(double_type()),ssReal.str(),o);
  EXPECT_EQ(1, count_matches("double", o.str()));

  ssReal.str(std::string());
  stan::lang::generate_real_var_type(fun_origin, true, ssReal);
  o.str(std::string());
  stan::lang::generate_bare_type(bare_expr_type(double_type()),ssReal.str(),o);
  EXPECT_EQ(1, count_matches("local_scalar_t__", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_real_var_type(fun_origin, false, ssReal);
  stan::lang::generate_bare_type(bare_expr_type(double_type()),ssReal.str(),o);
  EXPECT_EQ(1, count_matches("local_scalar_t__", o.str()));

  ssReal.str(std::string());
  o.str(std::string());
  stan::lang::generate_bare_type(bare_expr_type(int_type()), ssReal.str(), o);
  EXPECT_EQ(1, count_matches("int", o.str()));

  ssReal.str(std::string());
  stan::lang::generate_real_var_type(td_origin, false, ssReal);
  o.str(std::string());
  stan::lang::generate_bare_type(bare_expr_type(vector_type()),ssReal.str(),o);
  EXPECT_EQ(1, count_matches("Eigen::Matrix<double, Eigen::Dynamic, 1>", o.str()));

  ssReal.str(std::string());
  stan::lang::generate_real_var_type(td_origin, true, ssReal);
  o.str(std::string());
  stan::lang::generate_bare_type(bare_expr_type(vector_type()), ssReal.str(), o);
  EXPECT_EQ(1, count_matches("Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, 1>", o.str()));

  ssReal.str(std::string());
  stan::lang::generate_real_var_type(td_origin, false, ssReal);
  o.str(std::string());
  stan::lang::generate_bare_type(bare_expr_type(row_vector_type()), ssReal.str(), o);
  EXPECT_EQ(1, count_matches("Eigen::Matrix<double, 1, Eigen::Dynamic>", o.str()));
  ssReal.str(std::string());
  stan::lang::generate_real_var_type(td_origin, true, ssReal);
  o.str(std::string());
  stan::lang::generate_bare_type(bare_expr_type(row_vector_type()), ssReal.str(), o);
  EXPECT_EQ(1, count_matches("Eigen::Matrix<local_scalar_t__, 1, Eigen::Dynamic>", o.str()));

  ssReal.str(std::string());
  stan::lang::generate_real_var_type(td_origin, false, ssReal);
  o.str(std::string());
  stan::lang::generate_bare_type(bare_expr_type(matrix_type()), ssReal.str(), o);
  EXPECT_EQ(1, count_matches("Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>", o.str()));

  ssReal.str(std::string());
  stan::lang::generate_real_var_type(td_origin, true, ssReal);
  o.str(std::string());
  stan::lang::generate_bare_type(bare_expr_type(matrix_type()), ssReal.str(), o);
  EXPECT_EQ(1, count_matches("Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, Eigen::Dynamic>", o.str()));
}
