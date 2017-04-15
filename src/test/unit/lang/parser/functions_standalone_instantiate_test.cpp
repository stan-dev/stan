#include <gtest/gtest.h>
#include <sstream>
#include <test/test-models/good-standalone-functions/basic.hpp>

using namespace stan::math;

void expect_no_errors(const std::ostringstream& error_stream) {
  EXPECT_TRUE(error_stream.str().empty());  
}

TEST(lang_parser, functions_standalone_instantiate_double) {
  std::ostringstream error_stream;
  EXPECT_EQ(basic_functions::my_log1p_exp(5, &error_stream), log1p_exp(5)) 
    << "Problem instantiating a real -> real function from standalone compilation.";
  expect_no_errors(error_stream);

  basic_functions::vector_d my_vec(3);
  my_vec[0] = 0.1;
  my_vec[1] = 0.2;
  my_vec[2] = 15.786;
  EXPECT_EQ(basic_functions::my_vector_mul_by_5(my_vec, &error_stream), 
    multiply(my_vec,5.0))  
      << "Problem instantiating a vector -> vector function from standalone compilation.";
  expect_no_errors(error_stream);
}

