#include <gtest/gtest.h>
#include <stan/gm/compiler.hpp>

TEST(GmCompiler, compile) {
  std::stringstream msgs, stan_gm_in, cpp_out;
  std::string model_name = "m";
  bool include_main = true;
  std::string in_file_name = "input";
  

  stan_gm_in << "model { }";
  
  stan::gm::compile(&msgs, stan_gm_in, cpp_out, model_name, include_main, in_file_name);
  
  EXPECT_EQ("", msgs.str());
  EXPECT_EQ(std::string::npos, cpp_out.str().find("int main("));
}

TEST(GmCompiler, compile_include_main_false) {
  std::stringstream msgs, stan_gm_in, cpp_out;
  std::string model_name = "m";
  bool include_main = false;
  std::string in_file_name = "input";
  

  stan_gm_in << "model { }";
  
  stan::gm::compile(&msgs, stan_gm_in, cpp_out, model_name, include_main, in_file_name);
  
  EXPECT_EQ("", msgs.str());
  EXPECT_EQ(std::string::npos, cpp_out.str().find("int main("));
}
