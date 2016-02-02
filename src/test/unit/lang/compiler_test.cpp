#include <gtest/gtest.h>
#include <stan/lang/compiler.hpp>
#include <test/unit/util.hpp>

TEST(LangCompiler, compile) {
  std::stringstream msgs, stan_lang_in, cpp_out;
  std::string model_name = "m";
  

  stan_lang_in << "model { }";
  
  stan::lang::compile(&msgs, stan_lang_in, cpp_out, model_name);
  
  EXPECT_EQ("", msgs.str());
  EXPECT_EQ(std::string::npos, cpp_out.str().find("int main("));
}

TEST(LangCompiler, streams) {
  stan::test::capture_std_streams();

  std::stringstream msgs, stan_lang_in, cpp_out;
  std::string model_name = "m";

  stan_lang_in.str("model { }");
  EXPECT_NO_THROW(stan::lang::compile(0, stan_lang_in, cpp_out, model_name));

  stan_lang_in.str("model { }");
  std::stringstream out;
  EXPECT_NO_THROW(stan::lang::compile(&out, stan_lang_in, cpp_out, model_name));
  
  // TODO(carpenter): wrong place to test basic test util behavior
  // TODO(carpenter): functions failing have no doc
  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}
