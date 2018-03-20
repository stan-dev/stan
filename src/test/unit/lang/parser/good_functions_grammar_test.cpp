#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>

bool run_test(std::string& stan_code,
              std::stringstream& msgs) {
  bool pass = false;
  std::vector<stan::lang::function_decl_def> fdds;
  fdds = parse_functions(stan_code, pass, msgs);
  return pass;
}


TEST(Parser, parse_empty_functions_block) {
  std::string input("functions {\n"
                    "}\n");
  std::stringstream msgs;
  bool pass = run_test(input, msgs);
  EXPECT_TRUE(pass);
}

TEST(Parser, parse_fun1) {
  std::string input("functions {\n"
                    "  real fun1(real x)\n"
                    "  { return x; }\n"                    
                    "}\n");
  std::stringstream msgs;
  bool pass = run_test(input, msgs);
  EXPECT_TRUE(pass);
}

TEST(Parser, parse_fun2) {
  std::string input("functions {\n"
                    "  real fun2(data real x)\n"
                    "  { return x; }\n"                    
                    "}\n");
  std::stringstream msgs;
  bool pass = run_test(input, msgs);
  EXPECT_TRUE(pass);
}

TEST(Parser, parse_fun3) {
  std::string input("functions {\n"
                    "  real fun3(data real x)\n"
                    "  {\n"
                    "    int a;\n"                    
                    "    return x; }\n"                    
                    "}\n");
  std::stringstream msgs;
  bool pass = run_test(input, msgs);
  EXPECT_TRUE(pass);
}

TEST(Parser, parse_fun4) {
  std::string input("functions {\n"
                    "  real fun4(real x) {\n"
                    "    row_vector[7] d3_rv[3,4,5];\n"
                    "    return x;\n"
                    "  }\n"
                    "}\n");
  std::stringstream msgs;
  bool pass = run_test(input, msgs);
  EXPECT_TRUE(pass);
}

TEST(Parser, parse_harm_osc_ode) {
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
  std::stringstream msgs;
  bool pass = run_test(input, msgs);
  EXPECT_TRUE(pass);
}
