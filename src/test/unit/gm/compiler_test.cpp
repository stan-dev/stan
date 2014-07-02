#include <gtest/gtest.h>
#include <stan/gm/compiler.hpp>

TEST(GmCompiler, compile) {
  std::stringstream msgs, stan_gm_in, cpp_out;
  std::string model_name = "m";
  std::string in_file_name = "input";
  

  stan_gm_in << "model { }";
  
  stan::gm::compile(&msgs, stan_gm_in, cpp_out, model_name, in_file_name);
  
  EXPECT_EQ("", msgs.str());
  EXPECT_EQ(std::string::npos, cpp_out.str().find("int main("));
}
