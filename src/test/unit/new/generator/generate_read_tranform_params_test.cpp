#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <gtest/gtest.h>

// use block_var_decls grammar utility to generate list of bvds
// then check generated code

bool run_test(std::string& stan_code,
              std::stringstream& cpp_code) {

  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::block_var_decl> bvds;
  bvds = parse_var_decls(stan_code, pass, msgs);
  if (!pass) {
    cpp_code << msgs.str();
    return pass;
  }
  std::cout << "\ntest.stan:" << std::endl;
  std::cout << stan_code << std::endl;

  stan::lang::generate_read_transform_params(bvds, 1, cpp_code);

  std::cout << "test.hpp:" << std::endl;
  std::cout << cpp_code.str() << std::endl;

  return pass;
}

TEST(generateTransformInitsMethod, t1) {
  std::string input("real y;\n"
                    "real<lower=2.1, upper=2.9> z;\n");
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
}

TEST(generateTransformInitsMethod, t2) {
  std::string input("simplex[5] s5;\n"
                    "corr_matrix[5] cm5;\n");

  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS?
}

TEST(generateTransformInitsMethod, t3) {
  std::string input("real p_y;\n"
                    "real p_d4_y[2,3,4,5];\n"
                    "matrix[2,3] p_d2_mat[4,5];\n"
                    "corr_matrix[2] p_d2_corr_mat[4,5];\n"
                    "real<lower=0> sigma;\n");

  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS?
}
