#include <gtest/gtest.h>
#include <sstream>
#include <test/test-models/good-standalone-functions/basic.hpp>
#include <test/test-models/good-standalone-functions/special_functions.hpp>
#include <test/test-models/good-standalone-functions/integrate.hpp>

using namespace stan::math;

/**
 * Tests that the error stream is empty and resets the error stream.
 */
void expect_no_errors(std::ostringstream& error_stream) {
  EXPECT_TRUE(error_stream.str().empty()) << "Unexpected evaluation error: " << error_stream.str();  
  error_stream.str("");
  error_stream.clear();
}

TEST(lang_parser, functions_standalone_instantiate_double_basic) {
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

  EXPECT_EQ(basic_functions::int_only_multiplication(3, 8, &error_stream), 
    3 * 8)  
      << "Problem instantiating an int-only function from standalone compilation.";
  expect_no_errors(error_stream);
}

TEST(lang_parser, functions_standalone_instantiate_double_special) {
  double lp = 0.1;
  stan::math::accumulator<double> lp_accum;
  boost::ecuyer1988 rng(123456);
  std::ostringstream error_stream;

  special_functions_functions::test_lp(1.0, lp, lp_accum, &error_stream);
  expect_no_errors(error_stream);

  special_functions_functions::test_rng(1.1, rng, &error_stream);
  expect_no_errors(error_stream);

  special_functions_functions::test_lpdf(1.5, 6, &error_stream);
  expect_no_errors(error_stream);
}

TEST(lang_parser, functions_standalone_instantiate_numerical_integration) {
  std::ostringstream error_stream;

  //I am not interested in the results, just that the functions run
  integrate_functions::ode_integrate(&error_stream);
  expect_no_errors(error_stream);
}

