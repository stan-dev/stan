#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/grammars_utility.hpp>
#include <sstream>
#include <string>
#include <iostream>

TEST(Parser, parse_empty_functions_block) {
  std::string input("functions {\n"
                    "}\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fdds;
  fdds = parse_functions(input, pass, msgs);
  //  EXPECT_TRUE(pass);
  //  EXPECT_EQ(msgs.str(), std::string(""));
  std::cout << msgs.str() << std::endl;
}

TEST(Parser, parse_fun1) {
  std::string input("functions {\n"
                    "  real fun1(real x)\n"
                    "  { return x; }\n"                    
                    "}\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fdds;
  fdds = parse_functions(input, pass, msgs);
  //  EXPECT_TRUE(pass);
  //  EXPECT_EQ(msgs.str(), std::string(""));
  std::cout << msgs.str() << std::endl;
}

TEST(Parser, parse_fun2) {
  std::string input("functions {\n"
                    "  real fun2(data real x)\n"
                    "  { return x; }\n"                    
                    "}\n");
  bool pass = false;
  std::stringstream msgs;
  std::vector<stan::lang::function_decl_def> fdds;
  fdds = parse_functions(input, pass, msgs);
  //  EXPECT_TRUE(pass);
  //  EXPECT_EQ(msgs.str(), std::string(""));
  std::cout << msgs.str() << std::endl;
}
