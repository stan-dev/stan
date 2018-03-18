#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <gtest/gtest.h>

// use functions grammar utility to generate functions ast structures
// then check generated code

bool run_test(std::string& stan_code,
              std::stringstream& cpp_code) {

  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fdds;
  fdds = parse_functions(stan_code, pass, msgs);
  if (!pass) {
    cpp_code << msgs.str();
    return pass;
  }
  stan::lang::generate_functions(fdds, cpp_code);

  std::cout << "\ntest.stan:" << std::endl;
  std::cout << stan_code << std::endl;
  std::cout << "test.hpp:" << std::endl;
  std::cout << cpp_code.str() << std::endl;

  return pass;
}

TEST(generateFunctions, t1) {
  std::string input("functions {\n"
                    "  real fun1(real x)\n"
                    "  { return x; }\n"                    
                    "}\n");
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);

}

TEST(generateFunctions, t2) {
  std::string input("functions {\n"
                    "  real fun2(real x) {\n"
                    "    int a;\n"
                    "    matrix[2,2] i_mat = [ [ 1 , 0 ] , [ 0 , 1 ] ];\n"
                    "    row_vector[7] d3_rv[3,4,5];\n"
                    "    return x;\n"
                    "  }\n"
                    "}\n");
                    
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
}


TEST(generateFunctions, harm_osc_ode) {
  std::string input(
                    "functions {\n"
                    "  real[] harm_osc_ode(real t,\n"
                    "                      real[] y,         // state\n"
                    "                      real[] theta,     // parameters\n"
                    "                      real[] x,         // data\n"
                    "                      int[] x_int) {    // integer data\n"
                    "    real dydt[2];\n"
                    "    dydt[1] <- x[1] * y[2];\n"
                    "    dydt[2] <- -y[1] - theta[1] * y[2];\n"
                    "    return dydt;\n"
                    "  }\n"
                    "}\n"
                    );
  std::stringstream cpp_code;
  bool pass = run_test(input, cpp_code);
  EXPECT_TRUE(pass);
  std::cout << cpp_code.str() << std::endl;
}
