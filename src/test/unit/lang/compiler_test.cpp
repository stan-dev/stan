#include <gtest/gtest.h>
#include <stan/lang/compiler.hpp>

TEST(LangCompiler, compile) {
  std::stringstream msgs, stan_lang_in, cpp_out;
  std::string model_name = "m";
  std::string in_file_name = "input";
  

  stan_lang_in << "model { }";
  
  stan::lang::compile(&msgs, stan_lang_in, cpp_out, model_name, in_file_name);
  
  EXPECT_EQ("", msgs.str());
  EXPECT_EQ(std::string::npos, cpp_out.str().find("int main("));
}
