#include <stan/lang/generator.hpp>
#include <test/unit/lang/utility.hpp>
#include <gtest/gtest.h>
#include <sstream>

TEST(lang, generate_model_typedef) {
  std::string model_name = "name";
  std::stringstream ss;
  stan::lang::generate_model_typedef(model_name,ss);

  EXPECT_EQ(1, count_matches("typedef name_namespace::name stan_model;",
                             ss.str()));
}
