#include <stan/lang/ast_def.cpp>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>
#include <set>
#include <vector>

using stan::lang::expression;
using stan::lang::int_literal;
using stan::lang::function_signatures;
using stan::lang::bare_array_type;
using stan::lang::bare_expr_type;
using stan::lang::ill_formed_type;
using stan::lang::void_type;
using stan::lang::double_type;
using stan::lang::int_type;
using stan::lang::vector_type;
using stan::lang::row_vector_type;
using stan::lang::matrix_type;
using std::vector;


TEST(langAst, getDefinition) {
  stan::lang::function_signatures& fs
    = stan::lang::function_signatures::instance();
  std::string name = "f3args";
  bare_expr_type return_type = bare_expr_type(double_type());
  std::vector<bare_expr_type> arg_types;
  arg_types.push_back(bare_expr_type(bare_array_type(double_type(), 2U)));
  arg_types.push_back(bare_expr_type(bare_array_type(int_type(), 1U)));
  arg_types.push_back(bare_expr_type(vector_type()));

  // check is defined
  fs.add(name, return_type, arg_types);
  stan::lang::function_signature_t sig(return_type, arg_types);
  EXPECT_TRUE(fs.is_defined(name, sig));
}

TEST(langAst, missingDefinition) {
  stan::lang::function_signatures& fs
    = stan::lang::function_signatures::instance();

  std::string name = "fmissing";
  bare_expr_type return_type = bare_expr_type(double_type());
  std::vector<bare_expr_type> arg_types;
  arg_types.push_back(bare_expr_type(bare_array_type(double_type(), 2U)));


  stan::lang::function_signature_t sig(return_type, arg_types);
  EXPECT_FALSE(fs.is_defined(name, sig));
}

TEST(langAst, checkDefinition) {
  // tests for Stan lang function definitions with fun argument qualifier "data"
  stan::lang::function_signatures& fs
    = stan::lang::function_signatures::instance();
  std::string name = "f3args_data";

  bare_expr_type return_type = bare_expr_type(double_type());

  bare_expr_type data_only_2d_ar_double = bare_expr_type(bare_array_type(double_type(true), 2U));
  std::vector<bare_expr_type> arg_types;
  arg_types.push_back(data_only_2d_ar_double);
  EXPECT_TRUE(arg_types[0].is_data());
  arg_types.push_back(bare_expr_type(bare_array_type(int_type(), 1U)));
  arg_types.push_back(bare_expr_type(vector_type()));
  fs.add(name, return_type, arg_types);

  // check definition
  stan::lang::function_signature_t sig(return_type, arg_types);
  EXPECT_EQ(sig, fs.get_definition(name, sig));

  stan::lang::function_signature_t sig2 = fs.get_definition(name, sig);
  EXPECT_TRUE(sig2.first.is_double_type());
  EXPECT_EQ(sig2.second.size(), 3);

  // check function arguments
  EXPECT_TRUE(sig2.second[0].is_data());
  EXPECT_FALSE(sig2.second[1].is_data());
  EXPECT_FALSE(sig2.second[2].is_data());
}

TEST(langAst, discreteFirstArg) {
  // true if first argument to function is always discrete
  EXPECT_TRUE(function_signatures::instance()
              .discrete_first_arg("poisson_log"));
  EXPECT_FALSE(function_signatures::instance()
               .discrete_first_arg("normal_log"));
}

TEST(langAst, printSignature) {
  std::vector<bare_expr_type> arg_types;
  arg_types.push_back(bare_expr_type(bare_array_type(double_type(), 2U)));
  arg_types.push_back(bare_expr_type(bare_array_type(int_type(), 1U)));
  arg_types.push_back(bare_expr_type(vector_type()));
  std::string name = "foo";

  std::stringstream platform_eol_ss;
  platform_eol_ss << std::endl;
  std::string platform_eol = platform_eol_ss.str();

  std::stringstream msgs1;
  bool sampling_error_style1 = true;
  stan::lang::print_signature(name, arg_types, sampling_error_style1, msgs1);
  EXPECT_EQ("  real[ , ] ~ foo(int[ ], vector)" + platform_eol,
            msgs1.str());

  std::stringstream msgs2;
  bool sampling_error_style2 = false;
  stan::lang::print_signature(name, arg_types, sampling_error_style2, msgs2);
  EXPECT_EQ("  foo(real[ , ], int[ ], vector)" + platform_eol,
            msgs2.str());
  
  bare_expr_type bet_data_only = bare_expr_type(matrix_type(true));
  arg_types.push_back(bet_data_only);
  arg_types.push_back(bare_expr_type(matrix_type()));

  std::stringstream msgs3;
  stan::lang::print_signature(name, arg_types, sampling_error_style2, msgs3);
  EXPECT_EQ("  foo(real[ , ], int[ ], vector, data matrix, matrix)" + platform_eol,
            msgs3.str());

}
