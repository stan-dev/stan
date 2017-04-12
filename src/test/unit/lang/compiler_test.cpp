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
}

TEST(LangCompiler, equivEmpty) {
  std::stringstream msgs, empty_prog_in, empty_model_in, empty_prog_out,
    empty_model_out;
  std::string model_name = "m";

  empty_prog_in << "";
  stan::lang::compile(&msgs, empty_prog_in, empty_prog_out, model_name);

  empty_model_in << "model { }";
  stan::lang::compile(&msgs, empty_model_in, empty_model_out, model_name);

  // empty model contains extra newline, remove all for comparison
  std::string emptyprog(empty_prog_out.str());
  boost::algorithm::erase_all(emptyprog, "\n");
  std::string emptymodel(empty_model_out.str());
  boost::algorithm::erase_all(emptymodel, "\n");

  EXPECT_EQ(emptyprog, emptymodel);
}
