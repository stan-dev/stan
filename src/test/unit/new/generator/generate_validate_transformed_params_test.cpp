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

  stan::lang::generate_validate_transformed_params(bvds, 2, cpp_code);

  std::cout << "test.hpp:" << std::endl;
  std::cout << cpp_code.str() << std::endl;

  return pass;
}

TEST(generateValidateTransformedParams, t1) {
  std::string input("real y;\n"
                    "real<lower=2.1, upper=2.9> z;\n");
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
}

TEST(generateValidateTransformedParams, t2) {
  std::string input("matrix<lower=-7,upper=6>[3, 6] my_mat_mn;\n"
                    "matrix<upper=0>[2,4] d2_array_of_mat[8, 9];\n"
                    "matrix<upper=0>[2,4] d3_array_of_mat[7, 8, 9];\n");

  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS?
}

TEST(generateValidateTransformedParams, t5) {
  std::string input("simplex[5] s5;\n"
                    "corr_matrix[5] cm5;\n");

  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS?
}

TEST(generateValidateTransformedParams, t6) {
  std::string input(
                    "matrix[18,19] nb_mat;\n"
                    "matrix[18,19] ar_nb_mat[11];\n"
                    "matrix[18,19] d2_ar_nb_mat[11,12];\n"
                    "matrix[18,19] d3_ar_nb_mat[11,12,13];\n"
                    "matrix<lower = 0>[18,19] d3_ar_lb_mat[11,12,13];\n"
                    );
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS?
}
