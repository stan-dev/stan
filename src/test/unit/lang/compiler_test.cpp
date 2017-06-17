#include <gtest/gtest.h>
#include <stan/lang/compiler.hpp>
#include <test/unit/util.hpp>
#include <boost/algorithm/string.hpp>

TEST(LangCompiler, compile) {
  std::stringstream msgs, stan_lang_in, cpp_out;
  std::string model_name = "m";
  stan_lang_in << "model { }";

  stan::lang::compile(&msgs, stan_lang_in, cpp_out, model_name);
  EXPECT_EQ("", msgs.str());
  EXPECT_EQ(std::string::npos, cpp_out.str().find("int main("));
}

TEST(LangCompiler, emptyProgram) {
  std::stringstream msgs, stan_lang_in, cpp_out;
  std::string model_name = "m";
  stan_lang_in << "";

  stan::lang::compile(&msgs, stan_lang_in, cpp_out, model_name);

  EXPECT_EQ(1,count_matches("WARNING: empty program",msgs.str()));
  EXPECT_EQ(std::string::npos, cpp_out.str().find("int main("));

  // can't test equivalence of "" and "model { }" because of the
  // recording of the positions in the file
}
