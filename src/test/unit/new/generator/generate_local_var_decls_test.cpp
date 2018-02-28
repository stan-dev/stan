#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <gtest/gtest.h>

// use local_var_decls grammar utility to generate list of lvds
// then check generated code

bool run_test(std::string& stan_code,
              std::stringstream& cpp_code) {

  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::local_var_decl> lvds;
  lvds = parse_local_var_decls(stan_code, pass, msgs);
  if (!pass) {
    cpp_code << msgs.str();
    return pass;
  }
  std::cout << "\ntest.stan:" << std::endl;
  std::cout << stan_code << std::endl;

  stan::lang::generate_local_var_decl_inits(lvds, 1, cpp_code);

  std::cout << "test.hpp:" << std::endl;
  std::cout << cpp_code.str() << std::endl;

  return pass;
}

TEST(generateLocalVarInits, t1) {
  std::string input("int N;\n"
                    "real y;\n");
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);

}

TEST(generateLocalVarInits, t2) {
  std::string input("int N;\n"
                    "int x[N];\n"
                    "real y[N];\n");
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);

}

TEST(generateLocalVarInits, t3) {
  std::string input("int N;\n"
                    "int M;\n"
                    "int I;\n"
                    "int J;\n"
                    "matrix[M,N] my_mat_mn;\n"
                    "matrix[M, N] d1_array_of_mat[I];\n"
                    "matrix[M, N] d2_array_of_mat[I, J];\n");

  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS?
}

TEST(generateLocalVarInits, t4) {
  std::string input("vector[5] a;\n"
                    "vector[5] b[2];\n"
                    "vector[5] c[10,20,30];\n");
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS?
}

TEST(generateLocalVarInits, t5) {
  std::string input("row_vector[5] a;\n"
                    "row_vector[5] b[2];\n"
                    "row_vector[5] c[10,20,30];\n");
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  // TESTS?
}
